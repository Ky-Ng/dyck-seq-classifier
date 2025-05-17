import os
import torch
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

    model = Transformer.load_model(
        config_path,
        model_checkpoint
    )

    # pred = model.predict(torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1]).unsqueeze(0))
    pred = model.predict(torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(0))

    print(pred)


if __name__ == "__main__":
    main()
