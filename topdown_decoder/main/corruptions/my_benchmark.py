import os
from collections import defaultdict
from typing import Dict, Optional, List

import PIL.Image
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torchvision import transforms

from . import rgb_sensor_degradations
from .util import extract_patches, reconstruct_image

class corruptions:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
            self, visual_corruption, visual_severity, im_size=256
) -> None:
        visual_corruption, visual_severity = visual_corruption, visual_severity
        corrupted_patches_idx = 0

        if visual_corruption is not None and visual_severity > 0:  # This works
            self._corruptions = [visual_corruption.replace("_", " ")]
            self._severities = [visual_severity]
        else:
            self._corruptions = visual_corruption
            self._severities = visual_severity
        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

    def apply(self, frame):
        if isinstance(frame, PIL.Image.Image):
            im = np.array(frame)

        # Apply a sequence of corruptions to the RGB frames
        if self._corruptions is not None:
            im = rgb_sensor_degradations.apply_corruption_sequence(
                im, self._corruptions, self._severities
            )

        if not(isinstance(im, PIL.Image.Image)):
            im = self.to_pil(im)

        return im
