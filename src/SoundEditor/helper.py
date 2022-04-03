import numpy as np

from numpy.typing import NDArray
from enum import Enum

DATA_MODE = Enum("DATA_MODE", "ADD SUBTRACT REPLACE")

def bell_curve(halve_width: int) -> NDArray[np.float32]:
    if halve_width == 0:
        return np.array([1.0])
    else:
        x = np.arange(-halve_width, halve_width + 1, 1)
        return np.exp(-0.5 * (4.0 * x / halve_width) ** 2)  # 2.0 / (halve_width * np.sqrt(2.0 * np.pi)) *