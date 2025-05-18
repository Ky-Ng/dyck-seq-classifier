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
    parser.add_argument("-o", "--save_dir", required=False,
                        type=str, default=None, help="Directory to Save Heatmap")
    args = parser.parse_args()
    config_path, model_checkpoint, save_dir = args.config_path, args.model_checkpoint, args.save_dir

    interpeter = Interpretability(
        tokenizer=Tokenizer(symbols=["(", ")"]),
        model=Transformer.load_model(
            config_path,
            model_checkpoint
        ),
        save_dir=save_dir
    )

    interpeter.interpret("((((((()))))))")

if __name__ == "__main__":
    main()
