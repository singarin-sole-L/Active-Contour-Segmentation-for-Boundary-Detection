import argparse
import json
from pathlib import Path
import sys

import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from active_contour.preprocessing import load_grayscale
from active_contour.initialization import (
    initialize_circle,
    initialize_ellipse,
    initialize_from_bbox,
    auto_circle_initialization,
    auto_ellipse_initialization,
)
from active_contour.snake import ActiveContourSnake, SnakeConfig
from active_contour.metrics import evaluate_contour
from active_contour.visualization import plot_overlay, plot_evolution, plot_edge_map, save_contour_points


def parse_bbox(text):
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("bbox must be formatted as x,y,w,h")
    return values


def load_mask(path):
    arr = mpimg.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0.5


def build_initial_contour(args, image):
    h, w = image.shape

    if args.init == "circle":
        cx = args.cx if args.cx is not None else w / 2
        cy = args.cy if args.cy is not None else h / 2
        radius = args.radius if args.radius is not None else min(h, w) * 0.40
        return initialize_circle(cx, cy, radius, n_points=args.n_points)

    if args.init == "ellipse":
        cx = args.cx if args.cx is not None else w / 2
        cy = args.cy if args.cy is not None else h / 2
        rx = args.rx if args.rx is not None else w * 0.38
        ry = args.ry if args.ry is not None else h * 0.38
        return initialize_ellipse(cx, cy, rx, ry, n_points=args.n_points)

    if args.init == "bbox":
        if args.bbox is None:
            raise ValueError("--bbox x,y,w,h is required for bbox initialization")
        return initialize_from_bbox(*parse_bbox(args.bbox), n_points=args.n_points, scale=args.margin)

    if args.init == "auto_circle":
        return auto_circle_initialization(
            image,
            n_points=args.n_points,
            threshold=args.auto_threshold,
            margin=args.margin,
            polarity=args.polarity,
        )

    if args.init == "auto":
        return auto_ellipse_initialization(
            image,
            n_points=args.n_points,
            threshold=args.auto_threshold,
            margin=args.margin,
            polarity=args.polarity,
        )

    raise ValueError(f"Unknown init: {args.init}")


def main():
    parser = argparse.ArgumentParser(description="Run active contour segmentation on an image.")
    parser.add_argument("--input", required=True, help="Input grayscale or RGB image")
    parser.add_argument("--output_dir", required=True, help="Directory where outputs are saved")
    parser.add_argument("--mask", default=None, help="Optional ground truth mask for metrics")

    parser.add_argument("--init", default="auto", choices=["circle", "ellipse", "bbox", "auto", "auto_circle"])
    parser.add_argument("--polarity", default="auto", choices=["auto", "bright", "dark"], help="Object polarity")
    parser.add_argument("--bbox", default=None, help="x,y,w,h for bbox initialization")
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--rx", type=float, default=None)
    parser.add_argument("--ry", type=float, default=None)
    parser.add_argument("--n_points", type=int, default=160)
    parser.add_argument("--auto_threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=1.20)

    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.40)
    parser.add_argument("--gamma", type=float, default=3.50)
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--smoothing_sigma", type=float, default=1.5)
    parser.add_argument("--edge_power", type=float, default=2.0)
    parser.add_argument("--force_mode", default="mask_distance", choices=["mask_distance", "distance", "edge", "mixed"])
    parser.add_argument("--edge_threshold", type=float, default=0.25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--contour_smoothing_sigma", type=float, default=0.60)
    parser.add_argument("--invert", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_grayscale(args.input, invert=args.invert)
    initial = build_initial_contour(args, image)

    config = SnakeConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        dt=args.dt,
        iterations=args.iterations,
        smoothing_sigma=args.smoothing_sigma,
        edge_power=args.edge_power,
        force_mode=args.force_mode,
        polarity=args.polarity,
        auto_threshold=args.auto_threshold,
        edge_threshold=args.edge_threshold,
        save_every=args.save_every,
        contour_smoothing_sigma=args.contour_smoothing_sigma,
    )

    snake = ActiveContourSnake(config)
    result = snake.fit(image, initial)

    contour = result["contour"]
    history = result["history"]
    edge_map = result["edge_map"]

    plot_overlay(image, initial, title="Initial contour", save_path=output_dir / "initial_contour.png")
    plot_overlay(image, contour, title="Final contour", save_path=output_dir / "final_contour.png")
    plot_evolution(image, history, title="Snake evolution", save_path=output_dir / "snake_evolution.png")
    plot_edge_map(edge_map, save_path=output_dir / "edge_map.png")
    save_contour_points(contour, output_dir / "final_contour.csv")

    metrics = {}
    if args.mask:
        mask = load_mask(args.mask)
        metrics = evaluate_contour(contour, mask)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved results to {output_dir}")
    if metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
