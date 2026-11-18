import torch
from torch.nn import functional as F

from camus_seg.model import pixel_cross_entropy


def test_deterministic_ce_matches_reference_values_and_gradients():
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn((2, 4, 5, 7), generator=generator, dtype=torch.float64,
                         requires_grad=True)
    mask = torch.randint(4, (2, 5, 7), generator=generator)
    truth = F.one_hot(mask, 4).permute(0, 3, 1, 2).double()
    actual = pixel_cross_entropy(logits, truth)
    expected = F.cross_entropy(logits, mask)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
