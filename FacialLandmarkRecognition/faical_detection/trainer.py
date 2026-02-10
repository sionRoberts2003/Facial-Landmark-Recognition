import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

from config import FacialDetTrainerConfig

from performance_evaluator import Evaluator
from facial_detection import FacialDetection
from dataloader import FacialDetDataloader


class FacialDetectionTrainer:
    def __init__(self):
        self.model = FacialDetection()

        self.dataloader = FacialDetDataloader()
        print(self.dataloader.landmark_keys)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(),
                              lr=1e-4)

        print(f"Dataset length: {len(self.dataloader.training_dataset)}")

    def test(self):
        self.model.eval()
        for i, (_input, output) in enumerate(self.dataloader.validation_dataset):
            image = _input[0, 0]
            output = output[0] * 96

            model_output = self.model(_input)[0] * 96

            coordinates_label = [[output[i], output[i + 1]]
                                 for i in range(0, output.size(0), 2)]

            coordinates_label = torch.tensor(coordinates_label)

            coordinates_predicted = [[model_output[i], model_output[i + 1]]
                                     for i in range(0, model_output.size(0), 2)]
            coordinates_predicted = torch.tensor(coordinates_predicted)

            plt.imshow(image.detach().numpy(), cmap='grey')
            plt.scatter(coordinates_label[:, 0], coordinates_label[:, 1])
            plt.scatter(coordinates_predicted[:, 0], coordinates_predicted[:, 1])

            plt.savefig(f"../results/faces/face_sample_{i}.png")
            plt.close()

    def validate(self) -> float:
        avg_loss = 0

        for _input, output in self.dataloader.validation_dataset:
            model_output = self.model(_input)
            loss = self.criterion(model_output, output)

            avg_loss += loss.item() / (len(self.dataloader.validation_dataset))

        return avg_loss

    def train(self):
        loss_progression = []
        val_loss_progression = []

        for epoch in range(FacialDetTrainerConfig.EPOCH_COUNT):
            avg_loss = 0

            self.model.train()

            for i, (_input, output) in enumerate(self.dataloader.training_dataset):
                self.optimizer.zero_grad()

                model_output = self.model(_input)

                loss = self.criterion(model_output,
                                      output)
                loss.backward()

                avg_loss += loss.item() / (len(self.dataloader.training_dataset))
                self.optimizer.step()

            val_loss = self.validate()

            loss_progression.append(avg_loss)
            val_loss_progression.append(val_loss)
            print()
            print(f"Epoch: {epoch}")
            print(f"Average Loss: {avg_loss}")
            print(f"Average validation Loss: {val_loss}")
            print()

        torch.save(self.model, 'best_model.pt')
        self.plot_loss(loss_progression, val_loss_progression)

    def plot_loss(self, loss_progression: list[float], val_loss_progression: list[float]):
        plt.title("Loss Progression")
        plt.xlabel("Epochs")
        plt.ylabel("MSE loss values")
        plt.plot(loss_progression, label="Training")
        plt.plot(val_loss_progression, label="Validation")
        plt.legend()
        plt.savefig("../results/metrics/loss.png")
        plt.close()


if __name__ == '__main__':
    evaluator = Evaluator()

    trainer = FacialDetectionTrainer()
    trainer.train()
    trainer.test()

    evaluator.evaluate_model(torch.load("best_model.pt", weights_only=False),
                             trainer.dataloader.validation_dataset)
