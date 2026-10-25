"""
Contrast-Adaptive Potts Markov Random Field (MRF) and Lesion Morphometry Engine.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.ndimage import label, center_of_mass, find_objects


def icm_mrf_edge_aware(
    R: np.ndarray,
    t: float,
    scale: float,
    beta: float = 1.0,
    iters: int = 8
) -> np.ndarray:
    """
    Industry-Grade Edge-Preserving Contrast-Adaptive Potts MRF.
    Calculates spatial gradient of the residual map so smoothing respects anatomical borders.
    """
    if R.ndim == 2:
        R = R[None, ...]

    s = ((R - t) / scale).astype(np.float32)
    y = (s > 0).astype(np.float32)
    ii, jj = np.indices(R.shape[1:])
    masks = [((ii + jj) % 2 == k)[None] for k in (0, 1)]

    # Spatial gradient magnitude
    grad_y = np.abs(np.diff(R, axis=1, prepend=R[:, :1, :]))
    grad_x = np.abs(np.diff(R, axis=2, prepend=R[:, :, :1]))
    grad = np.sqrt(grad_y**2 + grad_x**2)
    sigma_e = np.std(grad) + 1e-6
    edge_w = np.exp(-grad / (2 * sigma_e)).astype(np.float32)

    def nbr(y_arr):
        p = np.pad(y_arr, ((0, 0), (1, 1), (1, 1)))
        return p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]

    for _ in range(iters):
        for m in masks:
            n1 = nbr(y)
            b = beta * edge_w
            E1 = -s + b * (4 - n1)
            E0 = s + b * n1
            y = np.where(m, (E1 < E0).astype(np.float32), y)

    return y.astype(np.uint8)


def analyze_lesions(
    mask: np.ndarray,
    img_shape: Tuple[int, int] = (128, 128)
) -> List[Dict[str, Any]]:
    """
    Automated Lesion Morphometry & Anatomical Zone Mapping.
    Returns cluster bounding boxes, centroids, pixel counts, and anatomical zones.
    """
    labeled, num_features = label(mask)
    slices = find_objects(labeled)
    lesions = []
    for i, s in enumerate(slices):
        c_mask = (labeled == (i + 1))
        p_cnt = int(c_mask.sum())
        if p_cnt < 4:
            continue
        cy, cx = center_of_mass(c_mask)
        ymin, ymax = s[0].start, s[0].stop
        xmin, xmax = s[1].start, s[1].stop

        side = "Right" if cx < img_shape[1] / 2 else "Left"
        if cy < img_shape[0] / 3:
            zone = f"{side} Upper Zone (RUZ/LUZ)"
        elif cy < 2 * img_shape[0] / 3:
            zone = f"{side} Mid Zone (RMZ/LMZ)"
        else:
            zone = f"{side} Lower Zone (RLZ/LLZ)"

        lesions.append({
            "cluster_id": i + 1,
            "anatomical_zone": zone,
            "centroid": (round(float(cx), 1), round(float(cy), 1)),
            "bbox": (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)),
            "pixel_count": p_cnt,
            "area_pct": round(float(p_cnt / (img_shape[0] * img_shape[1])) * 100, 2),
            "est_area_mm2": round(float(p_cnt * ((350.0 / img_shape[0]) ** 2)), 1)
        })
    return lesions
