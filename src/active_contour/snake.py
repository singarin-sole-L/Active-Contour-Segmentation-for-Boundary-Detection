from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d

from .preprocessing import compute_edge_map, gradient_field, distance_to_edges_force, mask_boundary_force


@dataclass
class SnakeConfig:
    alpha: float = 0.10
    beta: float = 0.40
    gamma: float = 3.50
    dt: float = 0.15
    iterations: int = 1500
    smoothing_sigma: float = 1.5
    edge_power: float = 2.0
    force_mode: str = "mask_distance"  # "mask_distance", "distance", "edge", "mixed"
    polarity: str = "auto"
    auto_threshold: float = 0.5
    edge_threshold: float = 0.25
    save_every: int = 100
    resample_every: int = 25
    contour_smoothing_sigma: float = 0.60


def _cyclic_second_matrix(n: int) -> np.ndarray:
    d2 = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        d2[i, i] = -2.0
        d2[i, (i - 1) % n] = 1.0
        d2[i, (i + 1) % n] = 1.0
    return d2


def _resample_closed_curve(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed contour to keep approximately uniform spacing."""
    pts = np.asarray(points, dtype=np.float32)
    closed = np.vstack([pts, pts[0]])
    diffs = np.diff(closed, axis=0)
    dist = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(dist)])
    total = s[-1]

    if total < 1e-8:
        return pts

    new_s = np.linspace(0, total, n_points + 1)[:-1]
    x = np.interp(new_s, s, closed[:, 0])
    y = np.interp(new_s, s, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _smooth_closed_curve(points: np.ndarray, sigma: float) -> np.ndarray:
    """Circular Gaussian smoothing of contour coordinates."""
    if sigma <= 0:
        return points
    x = gaussian_filter1d(points[:, 0], sigma=sigma, mode="wrap")
    y = gaussian_filter1d(points[:, 1], sigma=sigma, mode="wrap")
    return np.stack([x, y], axis=1).astype(np.float32)


class ActiveContourSnake:
    """
    Active contour model with several external force modes.

    Recommended for high-contrast black/white objects:
        force_mode="mask_distance"

    Recommended for natural grayscale images:
        force_mode="distance" or "mixed"
    """

    def __init__(self, config: SnakeConfig):
        self.config = config

    def _build_force_field(self, image):
        cfg = self.config

        if cfg.force_mode == "mask_distance":
            fx, fy, dist, edges, edge_map = mask_boundary_force(
                image,
                polarity=cfg.polarity,
                threshold=cfg.auto_threshold,
                smoothing_sigma=cfg.smoothing_sigma,
            )

        else:
            edge_map = compute_edge_map(
                image,
                smoothing_sigma=cfg.smoothing_sigma,
                edge_power=cfg.edge_power,
            )

            if cfg.force_mode == "edge":
                fx, fy = gradient_field(edge_map)
                dist = None
                edges = None

            elif cfg.force_mode == "distance":
                fx, fy, dist, edges = distance_to_edges_force(edge_map, edge_threshold=cfg.edge_threshold)

            elif cfg.force_mode == "mixed":
                fx_edge, fy_edge = gradient_field(edge_map)
                fx_dist, fy_dist, dist, edges = distance_to_edges_force(edge_map, edge_threshold=cfg.edge_threshold)
                fx = 0.35 * fx_edge + 0.65 * fx_dist
                fy = 0.35 * fy_edge + 0.65 * fy_dist
                norm = np.sqrt(fx * fx + fy * fy) + 1e-8
                fx = fx / norm
                fy = fy / norm

            else:
                raise ValueError("force_mode must be one of: mask_distance, edge, distance, mixed")

        return edge_map, fx.astype(np.float32), fy.astype(np.float32), dist, edges

    def fit(self, image: np.ndarray, initial_contour: np.ndarray):
        cfg = self.config
        h, w = image.shape

        edge_map, fx, fy, distance_map, edge_mask = self._build_force_field(image)

        contour = initial_contour.astype(np.float32).copy()
        n = contour.shape[0]

        d2 = _cyclic_second_matrix(n)
        d4 = d2 @ d2

        internal = cfg.alpha * d2 - cfg.beta * d4
        inv_matrix = np.linalg.inv(np.eye(n, dtype=np.float32) - cfg.dt * internal)

        history = [contour.copy()]
        energies = []

        for it in range(1, cfg.iterations + 1):
            x = np.clip(contour[:, 0], 0, w - 1)
            y = np.clip(contour[:, 1], 0, h - 1)

            external_fx = map_coordinates(fx, [y, x], order=1, mode="nearest")
            external_fy = map_coordinates(fy, [y, x], order=1, mode="nearest")

            new_x = inv_matrix @ (contour[:, 0] + cfg.dt * cfg.gamma * external_fx)
            new_y = inv_matrix @ (contour[:, 1] + cfg.dt * cfg.gamma * external_fy)

            contour = np.stack([new_x, new_y], axis=1).astype(np.float32)
            contour[:, 0] = np.clip(contour[:, 0], 0, w - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, h - 1)

            if cfg.contour_smoothing_sigma > 0 and it % 5 == 0:
                contour = _smooth_closed_curve(contour, cfg.contour_smoothing_sigma)

            if cfg.resample_every and it % cfg.resample_every == 0:
                contour = _resample_closed_curve(contour, n)

            if cfg.save_every and it % cfg.save_every == 0:
                history.append(contour.copy())

            if it % 20 == 0 or it == 1:
                energies.append(self._energy(contour, edge_map))

        if len(history) == 0 or not np.allclose(history[-1], contour):
            history.append(contour.copy())

        return {
            "contour": contour,
            "history": history,
            "edge_map": edge_map,
            "distance_map": distance_map,
            "edge_mask": edge_mask,
            "energies": energies,
        }

    def _energy(self, contour: np.ndarray, edge_map: np.ndarray) -> float:
        cfg = self.config
        x = contour[:, 0]
        y = contour[:, 1]

        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)

        internal = cfg.alpha * np.mean(dx * dx + dy * dy) + cfg.beta * np.mean(ddx * ddx + ddy * ddy)

        h, w = edge_map.shape
        xx = np.clip(x, 0, w - 1)
        yy = np.clip(y, 0, h - 1)
        edge_values = map_coordinates(edge_map, [yy, xx], order=1, mode="nearest")
        external = -cfg.gamma * np.mean(edge_values)

        return float(internal + external)
