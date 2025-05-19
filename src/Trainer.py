from dataclasses import dataclass
from enum import Enum, auto
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from Transformer import Transformer


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    num_epochs: int = 50
    verbose: bool = True
    positive_threshold: float = 0.75
    negative_threshold: float = 0.25
    save_dir: str = "./checkpoint/models"


class Trainer():
    class EvalMode(Enum):
        TRAIN = auto()
        VALID = auto()
        TEST = auto()

    def __init__(self, config: TrainerConfig, model: Transformer, x: torch.tensor, y: torch.tensor, val_x: torch.tensor, val_y: torch.tensor, test_x: torch.tensor = None, test_y: torch.tensor = None):
        # Hyperparameters
        self.learning_rate = config.learning_rate
        self.positive_threshold = config.positive_threshold
        self.negative_threshold = config.negative_threshold
        self.num_epochs = config.num_epochs
        self.save_dir = config.save_dir
        self.model = model

        # Data
        self.x = x
        self.y = y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate)

        self.verbose = config.verbose

        # Tracking Metrics
        self.train_accuracy = []
        self.valid_accuracy = []
        self.train_unconfidence = []
        self.valid_unconfidence = []
        self.losses = []

    def train(self):
        best_accuracy, best_unconfidence, best_epoch = -1.0, -1.0, -1
        for i in range(self.num_epochs):
            self.optimizer.zero_grad()
            logits, loss = self.model(self.x, self.y)
            loss.backward()  # Accumulates gradients
            self.optimizer.step()  # updates params

            # Loss is a single 1D tensor, item() moves back to CPU
            self.losses.append(loss.item())

            # Evaluate Train and Validation Loss
            self.model.set_is_training(False)
            predict_train = self.model.predict(self.x)
            predict_valid = self.model.predict(self.val_x)
            self.model.set_is_training(True)

            train_accuracy, train_unconfident = self.get_accuracy(
                predict_train, self.y)
            valid_accuracy, valid_unconfident = self.get_accuracy(
                predict_valid, self.val_y)

            # Logging Metrics
            self.train_accuracy.append(train_accuracy)
            self.train_unconfidence.append(train_unconfident)

            self.valid_accuracy.append(valid_accuracy)
            self.valid_unconfidence.append(valid_unconfident)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_unconfidence = valid_unconfident
                best_epoch = i
                self.model.save_model(self.save_dir)

            if self.verbose:
                print(
                    f"""step {i}:
                        loss: {loss.item()}
                        train_accuracy: {train_accuracy}
                        train_unconfidence: {train_unconfident}
                        valid_accuracy: {valid_accuracy}
                        valid_unconfident: {valid_unconfident}
                    """)

        print(
            f"Best Epoch = {best_epoch} with accuracy = {best_accuracy} and unconfidence = {best_unconfidence}")
        self.plot_training_curves()

    def eval(self, mode: EvalMode, verbose: bool = True) -> tuple[float, float]:
        match mode:
            case Trainer.EvalMode.TRAIN:
                return self._eval(self.x, self.y, "Train", verbose)
            case Trainer.EvalMode.VALID:
                return self._eval(self.val_x, self.val_y, "Valid", verbose)
            case Trainer.EvalMode.TEST:
                if self.test_x is None or self.test_y is None:
                    raise ValueError("Test data not provided.")
                return self._eval(self.test_x, self.test_y, "Test", verbose)
            case _:
                raise ValueError(f"Unsupported EvalMode: {mode}")

    def _eval(self, inputs: torch.tensor, labels: torch.tensor, split_name: str, verbose: bool) -> tuple[float, float]:
        self.model.set_is_training(False)
        preds = self.model.predict(inputs)
        self.model.set_is_training(True)

        accuracy, unconf = self.get_accuracy(preds, labels)
        if verbose:
            print(
                f"{split_name} Accuracy = {accuracy:.4f}, Unconfidence Rate = {unconf:.4f}")
        return accuracy, unconf

    def get_accuracy(self, predictions: torch.tensor, labels: torch.tensor) -> tuple[float, float]:
        """
        Converts probabilities into labels using the confidence (positive/negative) thresholds in self.config
        Marks unconfident probabilities and incorrect classifications as incorrect

        accuracy, percent_unconfident = get_accuracy(torch.tensor([0.8, 0.2, 0.3]), torch.tensor([1, 0, 0]))
        accuracy = 2/3
        percent_unconfident = 1/3 (0.3 is too high to be considered a confident negative for self.negative_threshold=0.25)
        """
        # Step 1) Construct masks for confident predictions
        confidence_1 = predictions >= self.positive_threshold
        confidence_0 = predictions <= self.negative_threshold

        # Step 2) Score confident_positive as 1, confident_negative as 0, and unconfident as sentinel value -1
        predicted_labels = torch.full_like(labels, -1.0, dtype=torch.float)
        predicted_labels[confidence_1] = 1
        predicted_labels[confidence_0] = 0

        # Step 3) Calculate mask of correct confident predictions
        labels = labels.float()  # Ensure labels is a float
        is_correct_mask = (predicted_labels == labels).float()

        # Step 4) Mark all unconfident values of -1.0 to 0.0 for averaging
        unconfident_mask = predicted_labels == -1.0
        is_correct_mask[unconfident_mask] = 0.0

        # Step 5) Return accuracy and percent unconfident
        accuracy = is_correct_mask.mean().item()
        percent_unconfident = unconfident_mask.float().mean().item()
        return accuracy, percent_unconfident

    def plot_training_curves(self):
        """
        Generated by GPT
        """
        epochs = list(range(1, len(self.losses) + 1))

        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.losses, label="Loss", color="black")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_accuracy, label="Train Accuracy")
        plt.plot(epochs, self.valid_accuracy, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.grid(True)
        plt.legend()

        # Plot unconfidence
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.train_unconfidence, label="Train Unconfidence")
        plt.plot(epochs, self.valid_unconfidence, label="Valid Unconfidence")
        plt.xlabel("Epoch")
        plt.ylabel("Unconfidence")
        plt.title("Unconfidence Rate")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
