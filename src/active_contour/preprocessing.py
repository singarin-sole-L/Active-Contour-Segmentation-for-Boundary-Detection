import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    distance_transform_edt,
    binary_fill_holes,
    binary_erosion,
    label,
)


def normalize01(image: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1]."""
    image = image.astype(np.float32)
    mn, mx = np.min(image), np.max(image)
    if mx - mn < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mn) / (mx - mn)


def load_grayscale(path: str, invert: bool = False) -> np.ndarray:
    """Load an image as grayscale float array in [0, 1]."""
    image = mpimg.imread(path)

    if image.ndim == 3:
        image = image[..., :3]
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    image = normalize01(image)

    if invert:
        image = 1.0 - image

    return image.astype(np.float32)


def salt_pepper_noise(image: np.ndarray, probability: float = 0.01, seed: int = 0) -> np.ndarray:
    """Apply salt-and-pepper noise to a grayscale image."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    r = rng.random(image.shape)
    noisy[r < probability] = 0.0
    noisy[r > 1.0 - probability] = 1.0
    return noisy


def foreground_mask(
    image: np.ndarray,
    polarity: str = "auto",
    threshold: float = 0.5,
    min_area: float = 0.005,
    max_area: float = 0.80,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Estimate the foreground object mask.

    polarity:
    - "bright": object is brighter than background
    - "dark": object is darker than background
    - "auto": choose the most plausible foreground among bright/dark candidates
    """
    image = normalize01(image)
    bright = image > threshold
    dark = image < threshold

    if polarity == "bright":
        mask = bright
    elif polarity == "dark":
        mask = dark
    elif polarity == "auto":
        total = image.size
        candidates = []
        for name, candidate in [("bright", bright), ("dark", dark)]:
            area = candidate.sum() / total
            if min_area <= area <= max_area:
                candidates.append((area, name, candidate))

        if not candidates:
            mask = dark if dark.sum() < bright.sum() else bright
        else:
            candidates.sort(key=lambda x: x[0])
            mask = candidates[0][2]
    else:
        raise ValueError("polarity must be one of: auto, bright, dark")

    if keep_largest:
        labels, n = label(mask)
        if n > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            largest = counts.argmax()
            mask = labels == largest

    if fill_holes:
        mask = binary_fill_holes(mask)

    return mask.astype(bool)


def compute_edge_map(
    image: np.ndarray,
    smoothing_sigma: float = 1.5,
    edge_power: float = 2.0,
) -> np.ndarray:
    """Compute a gradient-based edge attraction map."""
    smoothed = gaussian_filter(image, sigma=smoothing_sigma)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    grad_norm = np.sqrt(gx * gx + gy * gy)
    edge_map = normalize01(grad_norm) ** edge_power
    return edge_map.astype(np.float32)


def gradient_field(edge_map: np.ndarray):
    """Return gradients of an edge map along x and y."""
    fy, fx = np.gradient(edge_map)
    return fx.astype(np.float32), fy.astype(np.float32)


def distance_to_edges_force(
    edge_map: np.ndarray,
    edge_threshold: float = 0.25,
    percentile: float = 90.0,
):
    """
    Build a long-range external force from the distance transform to strong edges.
    """
    values = edge_map[edge_map > 0]

    if values.size > 0:
        adaptive = np.percentile(values, percentile) * 0.4
        threshold = max(edge_threshold, adaptive)
    else:
        threshold = edge_threshold

    edges = edge_map > threshold
    dist = distance_transform_edt(~edges).astype(np.float32)
    dist = normalize01(dist)

    fy, fx = np.gradient(dist)
    fx = -fx
    fy = -fy

    norm = np.sqrt(fx * fx + fy * fy) + 1e-8
    fx = fx / norm
    fy = fy / norm

    return fx.astype(np.float32), fy.astype(np.float32), dist.astype(np.float32), edges


def mask_boundary_force(
    image: np.ndarray,
    polarity: str = "auto",
    threshold: float = 0.5,
    smoothing_sigma: float = 1.0,
):
    """
    Build a force field from the outer boundary of the thresholded foreground object.

    This is the recommended mode for high-contrast objects such as a black droplet on a
    white background. It fills holes before extracting the boundary, so inner contours do not
    attract the snake when the goal is to detect the external object contour.
    """
    mask = foreground_mask(
        image,
        polarity=polarity,
        threshold=threshold,
        keep_largest=True,
        fill_holes=True,
    )

    eroded = binary_erosion(mask, iterations=1, border_value=0)
    boundary = mask ^ eroded

    if smoothing_sigma > 0:
        edge_map = gaussian_filter(boundary.astype(np.float32), sigma=smoothing_sigma)
        edge_map = normalize01(edge_map)
    else:
        edge_map = boundary.astype(np.float32)

    # Distance to the outer boundary.
    dist = distance_transform_edt(~boundary).astype(np.float32)
    dist = normalize01(dist)

    fy, fx = np.gradient(dist)
    fx = -fx
    fy = -fy

    norm = np.sqrt(fx * fx + fy * fy) + 1e-8
    fx = fx / norm
    fy = fy / norm

    return fx.astype(np.float32), fy.astype(np.float32), dist.astype(np.float32), boundary, edge_map
