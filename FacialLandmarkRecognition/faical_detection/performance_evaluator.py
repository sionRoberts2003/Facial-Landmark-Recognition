import math

import torch
from sympy.stats.rv_interface import standard_deviation
from torch.utils.data import DataLoader

from facial_detection import FacialDetection


class Evaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model: FacialDetection, data: DataLoader):
        mean, sd = self.get_point_deviation(model, data)
        print(f"mean error: {mean}")
        print(f"standard deviation of error: {sd}")

    def get_point_deviation(self, model: FacialDetection, data: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        raw_errors = self.get_errors(model, data)
        raw_errors = torch.tensor(raw_errors)
        mean = raw_errors.mean(dim=0)
        std = raw_errors.std(dim=0)

        return mean, std

    def get_errors(self, model: FacialDetection, data: DataLoader) -> list[list[float]]:
        return [
            self.point_error(
                self.get_coordinate_pairs(model(_input)[0]),
                actual_points=self.get_coordinate_pairs(output[0])
            )
            for _input, output in data
        ]

    def point_error(self,
                  predicted_points: list[tuple[float, float]],
                  actual_points: list[tuple[float, float]]) -> list[float]:
        return [
            math.pow(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2), 1/2)
            for (x1, y1), (x2, y2) in zip(predicted_points, actual_points)
        ]

    def get_coordinate_pairs(self, coordinates: torch.Tensor) -> list[tuple[float, float]]:
        return [(coordinates[i].item(), coordinates[i + 1].item())
                for i in range(0, coordinates.size(0), 2)]
