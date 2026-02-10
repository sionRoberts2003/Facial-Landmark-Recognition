import random
from multiprocessing.spawn import import_main_path

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from config import FacialDetTrainerConfig, FacialDetConfig


class FacialDetDataloader:
    def __init__(self):
        self.landmark_keys = {}

        self.images = self.load_images()
        self.images[torch.isnan(self.images)] = 0.0

        print(self.images.min())

        self.labels = self.load_labels()
        self.labels[torch.isnan(self.labels)] = 0.0

        total_data = list(zip(self.images, self.labels))
        random.shuffle(total_data)

        training_data = total_data[:int(len(total_data) * FacialDetTrainerConfig.TRAINING_PERCENT)]
        validation_data = total_data[len(training_data):]

        self.training_dataset = DataLoader(training_data,
                                           batch_size=FacialDetTrainerConfig.BATCH_SIZE,
                                           shuffle=True)
        self.validation_dataset = DataLoader(validation_data,
                                             batch_size=1)

    def load_labels(self) -> torch.Tensor:
        labels = pd.read_csv(FacialDetTrainerConfig.DATASET_LABELS_PATH)

        columns = [labels.columns[i]
                   for i in FacialDetTrainerConfig.ISOLATED_POINTS]

        self.landmark_keys = {c: i for i, c in enumerate(columns)}

        labels = labels[columns]
        label_tensor = torch.stack(
            [torch.from_numpy(labels.iloc[i].array.to_numpy())
             for i, _ in enumerate(labels.iloc)]
        )

        return label_tensor.float() / FacialDetConfig.IMG_HEIGHT

    def load_images(self) -> torch.Tensor:
        images = np.load(FacialDetTrainerConfig.DATASET_PATH)
        images = images[images.files[0]]

        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(1).float()
        max_values = [images[i].max()
                      for i, _ in enumerate(images)]
        max_values = torch.tensor(max_values)
        max_values = max_values.reshape(max_values.size(0), 1, 1, 1)

        return images / torch.tensor(max_values)


if __name__ == '__main__':
    loader = FacialDetDataloader()

    print(loader.landmark_keys)
