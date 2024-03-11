#!/usr/bin/env python3

import attr
import torch

from habitat_sim.registry import registry
from habitat_sim.sensor import SensorType
from habitat_sim.sensors.noise_models.sensor_noise_model import SensorNoiseModel


@registry.register_noise_model
@attr.s(auto_attribs=True, kw_only=True, slots=True)
class GaussianNoiseModelTorch(SensorNoiseModel):
    intensity_constant: float = 0.2
    mean: int = 0
    sigma: int = 1

    @staticmethod
    def is_valid_sensor_type(sensor_type: SensorType) -> bool:
        return sensor_type == SensorType.COLOR

    def simulate(self, image: torch.tensor) -> torch.tensor:
        noise = (torch.randn(image.shape[0], image.shape[1],
                             image.shape[2], device=image.device) * self.sigma + self.mean) * self.intensity_constant

        return (torch.maximum(torch.minimum(image / 255.0 + noise, torch.tensor(1.0, device=image.device)),
                              torch.tensor(0.0, device=image.device)) * 255.0)

    def apply(self, image: torch.tensor) -> torch.tensor:
        r"""Alias of `simulate()` to conform to base-class and expected API"""
        return self.simulate(image)
