\
import numpy as np
import matplotlib.pyplot as plt


def plot_overlay(image, contour, title="Final contour", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.plot(contour[:, 0], contour[:, 1], linewidth=2)
    ax.plot([contour[-1, 0], contour[0, 0]], [contour[-1, 1], contour[0, 1]], linewidth=2)
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=160)

    return fig


def plot_evolution(image, history, title="Snake evolution", save_path=None, max_curves=8):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")

    if len(history) <= max_curves:
        selected = history
    else:
        idx = np.linspace(0, len(history) - 1, max_curves).astype(int)
        selected = [history[i] for i in idx]

    for contour in selected:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=1.5, alpha=0.75)

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=160)

    return fig


def plot_edge_map(edge_map, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(edge_map, cmap="magma")
    ax.set_title("Edge attraction map")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=160)

    return fig


def save_contour_points(contour, path):
    np.savetxt(path, contour, delimiter=",", header="x,y", comments="")
