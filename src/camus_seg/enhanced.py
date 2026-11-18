"""Deeper U-Net, aspect-preserving geometry and explicit inference policies."""
import math

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch import nn
from torch.nn import functional as F

from .model import segmentation_loss


def norm_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.InstanceNorm2d(outputs, affine=True), nn.LeakyReLU(.01),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.InstanceNorm2d(outputs, affine=True), nn.LeakyReLU(.01))


class EnhancedUNet(nn.Module):
    """Four downsamplings, instance normalization, and native-resolution auxiliary heads."""
    def __init__(self, width=16):
        super().__init__()
        channels = [width * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([norm_block(1 if i == 0 else channels[i-1], c)
                                      for i, c in enumerate(channels)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i-1], 2, 2)
                                  for i in range(4, 0, -1)])
        self.decoders = nn.ModuleList([norm_block(channels[i-1]*2, channels[i-1])
                                       for i in range(4, 0, -1)])
        self.heads = nn.ModuleList([nn.Conv2d(channels[i], 4, 1) for i in range(3, -1, -1)])

    def forward(self, x):
        features = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else F.max_pool2d(x, 2))
            features.append(x)
        outputs = []
        for index, (up, decoder, head) in enumerate(zip(self.ups, self.decoders, self.heads)):
            x = decoder(torch.cat([up(x), features[-index-2]], 1))
            outputs.append(head(x))
        return tuple(reversed(outputs)) if self.training else outputs[-1]


def geometry(shape, size):
    height, width = shape
    scale = size / max(height, width)
    resized = max(1, round(height*scale)), max(1, round(width*scale))
    top, left = (size-resized[0])//2, (size-resized[1])//2
    return resized, (top, left)


def letterbox_input(image, size):
    image = torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
    lo, hi = torch.quantile(image, .01), torch.quantile(image, .99)
    image = ((image-lo)/(hi-lo).clamp_min(1e-6)).clamp(0, 1)
    shape, (top, left) = geometry(image.shape, size)
    resized = F.interpolate(image[None, None], shape, mode="bilinear", align_corners=False)[0]
    return F.pad(resized, (left, size-shape[1]-left, top, size-shape[0]-top))


def letterbox_mask(mask, size):
    shape, (top, left) = geometry(mask.shape, size)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, shape, mode="nearest")[0, 0].long()
    return F.pad(resized, (left, size-shape[1]-left, top, size-shape[0]-top))


def restore_logits(logits, original_shape):
    size = logits.shape[-1]
    (height, width), (top, left) = geometry(original_shape, size)
    crop = logits[..., top:top+height, left:left+width].float()
    return F.interpolate(crop, original_shape, mode="bilinear", align_corners=False)


def deep_supervision_loss(outputs, target):
    weights = [1., .5, .25, .125]
    total = outputs[0].new_zeros((), dtype=torch.float32)
    for output, weight in zip(outputs, weights):
        resized_target = F.interpolate(target[:, None].float(), output.shape[-2:],
                                       mode="nearest")[:, 0].long()
        total = total + weight*segmentation_loss(output.float(), resized_target)
    return total / sum(weights[:len(outputs)])


@torch.no_grad()
def affine_augment(images, masks):
    """Paired moderate affine transforms; no gradient through grid_sample is needed."""
    batch = len(images)
    angles = (torch.rand(batch, device=images.device)*20-10)*math.pi/180
    scales = torch.rand(batch, device=images.device)*.2 + .9
    theta = torch.zeros((batch, 2, 3), device=images.device)
    theta[:, 0, 0], theta[:, 1, 1] = angles.cos()/scales, angles.cos()/scales
    theta[:, 0, 1], theta[:, 1, 0] = -angles.sin()/scales, angles.sin()/scales
    # Normalized grid coordinates span [-1,1]: .16 corresponds to 8% of image extent.
    theta[:, :, 2] = torch.rand((batch, 2), device=images.device)*.32-.16
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = F.grid_sample(masks[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                          align_corners=False)[:, 0].long()
    return images, masks


def clean_components(prediction):
    """Retain one largest four-connected component per class; removed pixels become background."""
    result = np.zeros_like(prediction)
    for value in (1, 2, 3):
        components, count = connected_components(prediction == value)
        if count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            result[components == sizes.argmax()] = value
    return result


@torch.inference_mode()
def native_probabilities(model, tensor, original_shape, device, amp=True):
    model.eval()
    with torch.autocast(device_type=device, dtype=torch.float16,
                        enabled=amp and device == "cuda"):
        logits = model(tensor[None].to(device))
    return restore_logits(logits, original_shape).softmax(1)[0]
