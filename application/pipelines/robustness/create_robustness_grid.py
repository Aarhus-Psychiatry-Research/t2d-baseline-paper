from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH


def create_png_grid(input_folder: Path, output_folder: Path, g_height: int, g_width: int, fig_size: tuple[float, float]=(10,10), dpi: int=100) -> None:
    """Create a grid of PNG images from a folder of PNG images."""
    images: list[np.ndarray] = []
    input_folder = Path(input_folder)
    for filepath in input_folder.glob("*.png"):
        images.append(plt.imread(filepath))

    if len(images) == 0:
        raise ValueError("Folder does not contain any PNG files")

    fig, axs = plt.subplots(g_height, g_width, figsize=fig_size, dpi=dpi)
    fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(g_height):
        for j in range(g_width):
            index = i * g_width + j
            if index < len(images):
                axs[i,j].imshow(images[index], cmap="gray")
            axs[i,j].axis("off")

    fig.savefig(output_folder / "grid.png", dpi=dpi)



if __name__ == "__main__":
    create_png_grid(input_folder=ROBUSTNESS_PATH, output_folder=ROBUSTNESS_PATH, g_height=4, g_width=2, fig_size=(10,40), dpi=300)

