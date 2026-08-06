"""PyInstaller entry point for the windowed MCR-ALS application."""

import sys

import numpy as np

from mcr_als import MCRALSOptions, mcr_als
from mcr_als.gui import main


def self_test() -> None:
    """Exercise the bundled numerical libraries and Tk runtime."""
    concentrations = np.array(
        [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75], [0.0, 1.0]],
        dtype=np.float64,
    )
    spectra = np.array(
        [[1.0, 0.5, 0.1], [0.1, 0.5, 1.0]],
        dtype=np.float64,
    )
    result = mcr_als(
        concentrations @ spectra,
        spectra,
        MCRALSOptions(max_iterations=1),
    )
    if result.reconstructed_data.shape != (4, 3):
        raise RuntimeError("numerical self-test returned an unexpected shape")

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
