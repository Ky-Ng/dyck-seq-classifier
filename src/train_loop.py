import argparse
import subprocess
from Transformer import Transformer, TransformerConfig
from Trainer import Trainer, TrainerConfig
from DataLoader import DataLoader
from Tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--length", required=True,
                        type=int, help="Number of open parentheses")
    parser.add_argument("-e", "--epochs", required=True,
                        type=int, help="Number of epochs")

    args = parser.parse_args()
    n = args.length
    epochs = args.epochs

    subprocess.run(["python", "src/data_gen.py",
                   "-o", "data/input", "-n", f"{n}"])
    subprocess.run([
        "python", "src/split_data.py",
        "-fv", f"data/input/valid_parentheses_n{n}.txt",
        "-fi", f"data/input/invalid_parentheses_n{n}.txt",
        "-n", f"{n}",
        "-o", "./data/splits"
    ])

    # Load and Tokenize Data
    vocab = ["(", ")"]
    data_loader = DataLoader(
        tokenizer=Tokenizer(vocab),
        train_path=f"./data/splits/train_n{n}_unshuffled.csv",
        valid_path=f"./data/splits/valid_n{n}_unshuffled.csv",
        test_path=f"./data/splits/test_n{n}_unshuffled.csv"
    )

    x, y = data_loader.load_train()
    valid_x, valid_y = data_loader.load_valid()

    # Create Model
    config = TransformerConfig(
        block_size=2*n,
        n_head=4,
        n_embd=12
    )
    model = Transformer(config)

    # # Train Model
    trainer = Trainer(
        config=TrainerConfig(num_epochs=epochs),
        model=model,
        x=x,
        y=y,
        val_x=valid_x,
        val_y=valid_y
    )

    trainer.train()


if __name__ == "__main__":
    main()
