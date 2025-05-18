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
    parser.add_argument("-s", "--sequence", required=False,
                        type=str, default="((((((()))))))", help="Sequence to probe, defaults to `((((((()))))))`")
    parser.add_argument("-np", "--no-plot", dest="plot", action="store_false", help="Disable plotting")
    parser.set_defaults(plot=True) # Default if not specifying `-np` means we will plot
    
    args = parser.parse_args()
    config_path, model_checkpoint, save_dir, seq, show = args.config_path, args.model_checkpoint, args.save_dir, args.sequence, args.plot

    interpeter = Interpretability(
        tokenizer=Tokenizer(symbols=["(", ")"]),
        model=Transformer.load_model(
            config_path,
            model_checkpoint
        ),
        save_dir=save_dir,
        show=show
    )

    interpeter.interpret(seq)

if __name__ == "__main__":
    main()
