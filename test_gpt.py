from lib.Trainer import Trainer, TrainerConfig
from lib.GPT import GPT, GPTConfig
from lib.Tokenizer import Tokenizer
import torch
import argparse

print("Starting GPT Initialization")
parser = argparse.ArgumentParser(description="Test GPT in Dyck Language.")
parser.add_argument("-n", "--length", type=int, required=True, help="Number of pairs of parentheses")
args = parser.parse_args()
seq_len = args.length

VERBOSE = True
model = GPT(GPTConfig())
model.to("cpu")
vocab = ["(", ")"]
tokenizer = Tokenizer(vocab)


# TODO pad tokenizer with empty strings later


def load_data(path: str, num_open_paren: int) -> tuple:
    # Load in Data manually
    tokenized_corpus = []
    with open(path) as f:
        for line in f:
            tokens = tokenizer.encode(line.strip())
            tokenized_corpus.append(tokens)

        tokenized_corpus = torch.tensor(tokenized_corpus)
        print(tokenized_corpus)

        x = tokenized_corpus[:, :-1].contiguous().view(-1, num_open_paren*2)
        y = tokenized_corpus[:, 1:].contiguous().view(-1, num_open_paren*2)
        print("x", x)
        print("y", y)
    return (x, y)

# seq_len = 6
# tokenized_corpus = []
# # Load Training Data
# with open(f"data/splits/parentheses_n{seq_len}_train.txt") as f:
#     for line in f:
#         tokens = tokenizer.encode(line.strip())
#         tokenized_corpus.append(tokens)

# tokenized_corpus = torch.tensor(tokenized_corpus)
# print(tokenized_corpus)

# x = tokenized_corpus[:, :-1].contiguous().view(-1, seq_len*2)
# y = tokenized_corpus[:, 1:].contiguous().view(-1, seq_len*2)

x,y = load_data(f"data/splits/parentheses_n{seq_len}_train.txt", seq_len)
val_x, val_y = load_data(f"data/splits/parentheses_n{seq_len}_val.txt", seq_len)
print("x", x)
print("y", y)


trainer = Trainer(TrainerConfig(), model, x, y, val_x, val_y)
trainer.train()
trainer.validate()
