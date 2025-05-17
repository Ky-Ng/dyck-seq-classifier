import os
import torch
from Transformer import Transformer


def main():
    print("Initializing Model Probe")
    base_path = "./checkpoint/models/epoch156"

    model = Transformer.load_model(
        config_path=os.path.join(base_path, "transformer_n14_config.json"),
        model_checkpoint=os.path.join(base_path, "transformer_n14.pt")
    )

    # pred = model.predict(torch.tensor([0,1,0,1,0,1,0,1,0,1,0,1,0,1]).unsqueeze(0))
    pred = model.predict(torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(0))

    print(pred)


if __name__ == "__main__":
    main()
