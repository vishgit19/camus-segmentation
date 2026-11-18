import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    """Three-level U-Net with transposed convolutions and four exclusive classes."""
    def __init__(self, width=16):
        super().__init__()
        self.enc1, self.enc2 = block(1, width), block(width, width*2)
        self.middle = block(width*2, width*4)
        self.up2 = nn.ConvTranspose2d(width*4, width*4, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(width*2, width*2, 2, stride=2)
        self.dec2, self.dec1 = block(width*6, width*2), block(width*3, width)
        self.head = nn.Conv2d(width, 4, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([self.up2(c), b], dim=1))
        e = self.dec1(torch.cat([self.up1(d), a], dim=1))
        return self.head(e)


def input_tensor(image, size):
    image = torch.from_numpy(image.copy()).float()
    lo, hi = torch.quantile(image, .01), torch.quantile(image, .99)
    image = ((image-lo)/(hi-lo).clamp_min(1e-6)).clamp(0, 1)
    return F.interpolate(image[None, None], (size, size), mode="bilinear",
                         align_corners=False)[0]


def pixel_cross_entropy(logits, target_one_hot):
    # Equivalent to unweighted CE, avoiding CUDA's nondeterministic NLLLoss2d kernel.
    return -(target_one_hot * logits.log_softmax(1)).sum(1).mean()


def segmentation_loss(logits, mask):
    probability = logits.softmax(1)[:, 1:]
    all_truth = F.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    truth = all_truth[:, 1:]
    axes = (0, 2, 3)
    dice = (2*(probability*truth).sum(axes)+1e-6)/(probability.sum(axes)
            + truth.sum(axes)+1e-6)
    return .5*pixel_cross_entropy(logits, all_truth) + .5*(1-dice.mean())


@torch.inference_mode()
def predict(model, image, size, device):
    return predict_tensor(model, input_tensor(image, size), image.shape, device)


@torch.inference_mode()
def predict_tensor(model, tensor, original_shape, device):
    model.eval()
    logits = model(tensor[None].to(device))
    logits = F.interpolate(logits, original_shape, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()
