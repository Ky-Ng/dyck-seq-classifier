from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    num_epochs: int = 50
    verbose: bool = True


class Trainer():
    def __init__(self, config: TrainerConfig, model, x, y, val_x=None, val_y=None):
        # Hyperparameters
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.model = model

        # Data
        self.x = x
        self.y = y
        self.val_x = val_x
        self.val_y = val_y

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        self.verbose = config.verbose

    def train(self):
        for i in range(self.num_epochs):
            self.optimizer.zero_grad()
            logits, loss = self.model(self.x, self.y)
            loss.backward()  # Accumulates gradients
            self.optimizer.step()  # updates params

            # Loss is a single 1D tensor, item() moves back to CPU
            if self.verbose:
                print(f"step {i}, loss: {loss.item()}")

    def validate(self):
        if self.val_x is None or self.val_y is None:
            print(f"WARNING: Either val_x or val_y is none; unable to validate")
            return

        self.model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            logits = self.model(self.val_x)  # B, T, Vocab Size
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, VocabSize)
                self.val_y.view(-1),  # (B*T)
                reduction="sum"
            )
            total_loss += loss.item()
            count += self.val_x.numel()  # get loss per token

        val_loss_per_token = total_loss/count
        print(
            f"Validation Loss Per Token (total {count} tokens): ", val_loss_per_token)
        return val_loss_per_token
