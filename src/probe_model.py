from Interpretability import Interpretability
from Tokenizer import Tokenizer
from Transformer import Transformer
import argparse


def main():
    print("Initializing Model Probe")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True,
                        type=str, help="Path to Model Config")
    parser.add_argument("-m", "--model_checkpoint", required=True,
                        type=str, help="Path to Model Checkpoint")
    args = parser.parse_args()
    config_path, model_checkpoint = args.config_path, args.model_checkpoint

    interpeter = Interpretability(
        tokenizer=Tokenizer(symbols=["(", ")"]),
        model=Transformer.load_model(
            config_path,
            model_checkpoint
        )
    )

    interpeter.interpret("((((((()))))))")

if __name__ == "__main__":
    main()
