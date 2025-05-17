import torch
from Tokenizer import Tokenizer
import pandas as pd


class DataLoader():
    def __init__(self, tokenizer: Tokenizer, train_path: str, valid_path: str, test_path: str = None):
        self.tokenizer = tokenizer
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

    def load(self):
        """
        Returns as tuples of training, validation, and test data
        """
        return self.load_data(self.train_path), self.load_data(self.valid_path), self.load_data(self.test_path)

    def load_train(self):
        """
        x, y = data_loader.load_train()
        """
        return self.load_data(self.train_path)
    
    def load_valid(self):
        """
        valid_x, valid_y = data_loader.load_valid()
        """
        return self.load_data(self.valid_path)
    
    def load_test(self):
        """
        test_x, test_y = data_loader.load_test()
        """
        return self.load_data(self.test_path)

    def load_data(self, path: str) -> tuple[torch.tensor, torch.tensor]:
        """
        Loads inputs of shape (B,T) and applies tokenization
        Loads in label of shape (B,1)

        Returns:
            inputs, labels
        """
        df = pd.read_csv(path, skipinitialspace=True)
        df["tokenized"] = df["dyck_word"].apply(self.tokenizer.encode)
        df["grammatical"] = df["grammatical"].astype(float)

        inputs = torch.tensor(df["tokenized"].to_list(), dtype=torch.long)
        labels = torch.tensor(df["grammatical"].tolist(), dtype=torch.float).unsqueeze(1)
        return inputs, labels
