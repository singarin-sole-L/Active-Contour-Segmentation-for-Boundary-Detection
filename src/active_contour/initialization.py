import numpy as np

from .preprocessing import foreground_mask


def initialize_circle(center_x: float, center_y: float, radius: float, n_points: int = 160) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def initialize_ellipse(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    n_points: int = 160,
) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center_x + radius_x * np.cos(theta)
    y = center_y + radius_y * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def initialize_from_bbox(x: float, y: float, w: float, h: float, n_points: int = 160, scale: float = 1.15):
    """
    Initialize an ellipse from a bounding box.

    x, y = top-left corner
    w, h = width and height
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    rx = (w / 2.0) * scale
    ry = (h / 2.0) * scale
    return initialize_ellipse(cx, cy, rx, ry, n_points=n_points)


def auto_ellipse_initialization(
    image,
    n_points: int = 160,
    threshold: float = 0.5,
    margin: float = 1.20,
    polarity: str = "auto",
):
    """
    Estimate a foreground object and initialize an enclosing ellipse around it.

    This works for both:
    - bright object on dark background
    - dark object on bright background
    """
    mask = foreground_mask(image, polarity=polarity, threshold=threshold)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        h, w = image.shape
        return initialize_ellipse(w / 2, h / 2, w * 0.35, h * 0.35, n_points=n_points)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(4.0, (x1 - x0 + 1) / 2.0 * margin)
    ry = max(4.0, (y1 - y0 + 1) / 2.0 * margin)

    return initialize_ellipse(cx, cy, rx, ry, n_points=n_points)


def auto_circle_initialization(
    image,
    n_points: int = 160,
    threshold: float = 0.5,
    margin: float = 1.20,
    polarity: str = "auto",
):
    """
    Estimate a foreground object and initialize an enclosing circle around it.
    """
    mask = foreground_mask(image, polarity=polarity, threshold=threshold)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        h, w = image.shape
        return initialize_circle(w / 2, h / 2, min(h, w) * 0.35, n_points=n_points)

    cx = float((xs.min() + xs.max()) / 2.0)
    cy = float((ys.min() + ys.max()) / 2.0)
    radius = float(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.5 * margin)
    return initialize_circle(cx, cy, radius, n_points=n_points)
