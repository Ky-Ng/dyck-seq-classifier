import argparse
import subprocess
from Transformer import Transformer, TransformerConfig
from Trainer import Trainer, TrainerConfig
from DataLoader import DataLoader
from Tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True,
                        type=str, help="Path to Model Config")
    parser.add_argument("-m", "--model_checkpoint", required=True,
                        type=str, help="Path to Model Checkpoint")
    parser.add_argument("-n", "--length", required=True,
                        type=int, help="Number of open parentheses")
    args = parser.parse_args()
    n, config_path, model_checkpoint = args.length, args.config_path, args.model_checkpoint

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
    test_x, test_y = data_loader.load_test()

    trainer = Trainer(
        config=TrainerConfig(),  # Not needed
        model=Transformer.load_model(
            config_path,
            model_checkpoint
        ),
        x=x, y=y,
        val_x=valid_x, val_y=valid_y,
        test_x=test_x, test_y=test_y
    )

    trainer.eval(Trainer.EvalMode.TRAIN)
    trainer.eval(Trainer.EvalMode.VALID)
    trainer.eval(Trainer.EvalMode.TEST)


if __name__ == "__main__":
    main()
