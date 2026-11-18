import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_metrics(prediction, target, spacing_yx):
    """Four-connected pixel boundaries; distances use original-grid millimetres."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Metrics require matching 2D masks")
    spacing = np.asarray(spacing_yx, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Expected finite positive (y,x) spacing")
    p, t = int(prediction.sum()), int(target.sum())
    intersection = int((prediction & target).sum())
    result = dict(dice=2*intersection/(p+t) if p+t else 1.0,
                  iou=intersection/(p+t-intersection) if p+t else 1.0,
                  empty_prediction=p == 0, empty_target=t == 0,
                  surface_failure=(p == 0) != (t == 0))
    if p == 0 or t == 0:
        result.update(hd95_mm=None if p+t else 0.0, assd_mm=None if p+t else 0.0)
        return result
    p_boundary = prediction ^ binary_erosion(prediction, border_value=0)
    t_boundary = target ^ binary_erosion(target, border_value=0)
    p_to_t = distance_transform_edt(~t_boundary, sampling=spacing)[p_boundary]
    t_to_p = distance_transform_edt(~p_boundary, sampling=spacing)[t_boundary]
    result.update(hd95_mm=float(max(np.percentile(p_to_t, 95), np.percentile(t_to_p, 95))),
                  assd_mm=float(np.concatenate([p_to_t, t_to_p]).mean()))
    return result


def bootstrap_mean(values, repetitions=2000, seed=42):
    """Percentile bootstrap over patient summaries, never frames or pixels."""
    values = np.asarray(values, float)
    if values.size == 0 or not np.isfinite(values).all() or repetitions < 100:
        raise ValueError("Need finite patient values and at least 100 resamples")
    mean = float(values.mean())
    if len(values) < 2:
        return dict(mean=mean, ci95=None, n_patients=len(values))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)]
    return dict(mean=mean, ci95=np.percentile(means, [2.5, 97.5]).tolist(),
                n_patients=len(values))


def aggregate(rows, repetitions=2000, seed=42):
    result = {}
    for metric in ("dice", "iou", "hd95_mm", "assd_mm"):
        values, excluded = [], 0
        for patient in sorted({row["patient"] for row in rows}):
            all_values = [r[metric] for r in rows if r["patient"] == patient]
            valid = [v for v in all_values if v is not None]
            excluded += len(all_values) - len(valid)
            if valid:
                values.append(float(np.mean(valid)))
        result[metric] = bootstrap_mean(values, repetitions, seed) if values else {
            "mean": None, "ci95": None, "n_patients": 0}
        result[metric]["excluded_frame_class_pairs"] = excluded
    result["n_frame_class_pairs"] = len(rows)
    result["surface_failures"] = sum(r["surface_failure"] for r in rows)
    result["empty_targets"] = sum(r["empty_target"] for r in rows)
    result["empty_predictions"] = sum(r["empty_prediction"] for r in rows)
    return result
