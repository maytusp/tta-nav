import os
from collections import defaultdict
from typing import Dict, Optional, List

import PIL.Image
import imageio
import numpy as np
import torch
import tqdm
from habitat.config.default import get_agent_config, get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torchvision import transforms

from corruptions import rgb_sensor_degradations
from .util import extract_patches, reconstruct_image

class MyBenchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
            self, config
) -> None:
        visual_corruption, visual_severity = config.robustness.visual_corruption, config.robustness.visual_severity
        agent_config = get_agent_config(config.habitat.simulator)
        self.width, self.height = agent_config.sim_sensors.rgb_sensor.width, agent_config.sim_sensors.rgb_sensor.height
        self.patch_size = self.width
        corrupted_patches_idx = 0
        try:
            self.apply_local = config.robustness.apply_local_patch
        except:
            self.apply_local = False
        if visual_corruption is not None and visual_severity > 0:  # This works
            self._corruptions = [visual_corruption.replace("_", " ")]
            self._severities = [visual_severity]
        else:
            self._corruptions = visual_corruption
            self._severities = visual_severity
        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

        if self.apply_local:
            print("Apply Corruption on a Patch")
            self.patch_size = config.robustness.patch_size
            self.reset_corrupted_patches_idx()

    def reset(self):
        assert self.width == self.height, f"Only work when width and height are equal, the current ones are {self.width}, {self.height}"
        num_patches = int((self.width // self.patch_size) * (self.height // self.patch_size))
        self.corrupted_patches_idx = np.random.randint(0,num_patches,1)[0]

    def corrupt_rgb_observation(self, frame):
        if not(self.apply_local):
            # Work with numpy
            if type(frame) == torch.Tensor:
                im = frame.cpu().numpy().astype(np.uint8)
            else:
                im = np.array(frame)

            # Apply a sequence of corruptions to the RGB frames
            if self._corruptions is not None:
                im = rgb_sensor_degradations.apply_corruption_sequence(
                    im, self._corruptions, self._severities
                )

            if isinstance(im, PIL.Image.Image):
                im = np.array(im)

            # Return the same type
            if type(frame) == torch.Tensor:
                return torch.tensor(im, dtype=frame.dtype).to(frame.device)
            else:
                return im if isinstance(im, np.ndarray) else np.array(im)
        else:
            # Local corruption: Apply transformation for some image patches and keep other the sames
            # Work with numpy
            if type(frame) == torch.Tensor:
                im = frame.cpu().numpy().astype(np.uint8)
            else:
                im = np.array(frame)

            # Apply a sequence of corruptions to the RGB frames
            if self._corruptions is not None:
                # im.shape = [256,256,3]
                im_patches = extract_patches(im, self.width, self.patch_size)
                # print("im_patches shape", im_patches.shape)
                im_corrupted = im_patches[self.corrupted_patches_idx]
                im_corrupted = rgb_sensor_degradations.apply_corruption_sequence(
                    im_corrupted, self._corruptions, self._severities
                )
                im_patches[self.corrupted_patches_idx] = im_corrupted
                im = reconstruct_image(im_patches, self.width)

            if isinstance(im, PIL.Image.Image):
                im = np.array(im)


            # Return the same type
            if type(frame) == torch.Tensor:
                return torch.tensor(im, dtype=frame.dtype).to(frame.device)
            else:
                return im if isinstance(im, np.ndarray) else np.array(im)