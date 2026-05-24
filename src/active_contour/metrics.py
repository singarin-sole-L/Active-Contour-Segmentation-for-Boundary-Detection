import numpy as np
from matplotlib.path import Path
from scipy.spatial.distance import directed_hausdorff


def contour_to_mask(contour: np.ndarray, shape) -> np.ndarray:
    """Rasterize a closed contour into a binary mask."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    points = np.vstack((xx.ravel(), yy.ravel())).T
    closed_contour = np.vstack([contour, contour[0]])
    path = Path(closed_contour, closed=True)
    mask = path.contains_points(points).reshape(h, w)
    return mask


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(inter / (union + 1e-8))


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    inter = np.logical_and(pred, true).sum()
    return float(2 * inter / (pred.sum() + true.sum() + 1e-8))


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Approximate boundary points from a binary mask."""
    mask = mask.astype(bool)
    up = np.roll(mask, 1, axis=0)
    down = np.roll(mask, -1, axis=0)
    left = np.roll(mask, 1, axis=1)
    right = np.roll(mask, -1, axis=1)
    boundary = mask & (~(up & down & left & right))
    ys, xs = np.where(boundary)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def hausdorff_distance(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    a = mask_boundary_points(pred_mask)
    b = mask_boundary_points(true_mask)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def evaluate_contour(contour: np.ndarray, true_mask: np.ndarray):
    pred_mask = contour_to_mask(contour, true_mask.shape)
    return {
        "iou": iou_score(pred_mask, true_mask),
        "dice": dice_score(pred_mask, true_mask),
        "hausdorff": hausdorff_distance(pred_mask, true_mask),
    }
