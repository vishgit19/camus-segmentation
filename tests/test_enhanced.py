import numpy as np
import pytest
import torch
from torch.nn import functional as F

from camus_seg.common import write_json
from camus_seg.enhanced import (EnhancedUNet, affine_augment, clean_components,
                                deep_supervision_loss, geometry, letterbox_mask, restore_logits)
from camus_seg.improve import paired_comparison, select_candidate


def test_deep_supervision_shapes_gradients_and_eval_head():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EnhancedUNet(2)
    image = torch.randn(2, 1, 32, 32)
    outputs = model(image)
    assert [output.shape[-2:] for output in outputs] == [(32, 32), (16, 16), (8, 8), (4, 4)]
    target = torch.randint(4, (2, 32, 32))
    loss = deep_supervision_loss(outputs, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
               for head in model.heads)
    model.eval()
    torch.testing.assert_close(model(image), outputs[0].detach())


def test_letterbox_crop_restores_native_coordinates():
    target = np.zeros((16, 32), dtype=np.uint8)
    target[2:12, 7:23] = 2
    prepared = letterbox_mask(target, 32)
    assert geometry(target.shape, 32) == ((16, 32), (8, 0))
    assert prepared[:8].sum() == prepared[24:].sum() == 0
    logits = F.one_hot(prepared, 4).permute(2, 0, 1)[None].float()*20
    restored = restore_logits(logits, target.shape).argmax(1)[0].numpy()
    np.testing.assert_array_equal(restored, target)


def test_component_cleanup_is_classwise_and_idempotent():
    target = np.zeros((20, 20), dtype=np.uint8)
    target[2:8, 2:8] = 1
    target[15, 15] = 1
    target[10:18, 1:7] = 2
    target[2, 18] = 3
    cleaned = clean_components(target)
    assert cleaned[15, 15] == 0
    assert (cleaned == 1).sum() == 36
    assert (cleaned == 2).sum() == 48
    assert cleaned[2, 18] == 3
    np.testing.assert_array_equal(clean_components(cleaned), cleaned)
    np.testing.assert_array_equal(clean_components(np.zeros_like(target)), np.zeros_like(target))


def test_affine_pairs_are_seeded_and_aligned():
    mask = torch.zeros(3, 64, 64, dtype=torch.long)
    mask[:, 16:48, 16:48] = 1
    image = mask[:, None].float()
    torch.manual_seed(42)
    transformed, labels = affine_augment(image, mask)
    torch.manual_seed(42)
    repeated, repeated_labels = affine_augment(image, mask)
    torch.testing.assert_close(transformed, repeated)
    torch.testing.assert_close(labels, repeated_labels)
    assert set(labels.unique().tolist()) <= {0, 1}
    assert (transformed[:, 0].round() == labels).float().mean() > .99
    assert not torch.equal(labels[0], labels[1])


def test_selection_uses_predeclared_validation_rule():
    def candidate(name, dice, distance, failures=0):
        return dict(id=name, members=[0], validation=dict(dice=dice, hd95_mm=distance,
                                                          empty_mask_failures=failures))
    high_dice = candidate("high_dice", .901, 8.)
    low_boundary_error = candidate("low_boundary_error", .9005, 4.)
    undefined = candidate("undefined", .9008, 1., 1)
    assert select_candidate([high_dice, low_boundary_error, undefined])["id"] == "low_boundary_error"
    assert select_candidate([candidate("outside_tolerance", .89, 1.), high_dice])["id"] == "high_dice"


def test_paired_bootstrap_preserves_patient_pairs(tmp_path):
    baseline = {p: {m: dict(mean=.5) for m in ("dice", "iou", "hd95_mm", "assd_mm")}
                for p in ("a", "b", "c")}
    new = {p: {m: dict(mean=.6) for m in ("dice", "iou", "hd95_mm", "assd_mm")}
           for p in baseline}
    path = tmp_path / "baseline.json"
    write_json(path, baseline)
    result = paired_comparison(new, path, 100)
    assert result["dice"]["mean"] == pytest.approx(.1)
    assert result["dice"]["ci95"] == pytest.approx([.1, .1])
    assert result["dice"]["n_patients"] == 3
    with pytest.raises(ValueError, match="membership"):
        paired_comparison({"a": new["a"]}, path, 100)
