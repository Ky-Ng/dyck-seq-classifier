from Transformer import Transformer, TransformerConfig
from Trainer import Trainer, TrainerConfig
from DataLoader import DataLoader
from Tokenizer import Tokenizer

# Load and Tokenize Data
vocab = ["(", ")"]
data_loader = DataLoader(
    tokenizer=Tokenizer(vocab),
    train_path="./data/splits/train_n12_unshuffled.csv",
    valid_path="./data/splits/valid_n12_unshuffled.csv",
    test_path="./data/splits/test_n12_unshuffled.csv"
)

x, y = data_loader.load_train()
valid_x, valid_y = data_loader.load_valid()


# Create Model
config = TransformerConfig()
model = Transformer(config)

# # Train Model
trainer = Trainer(
    config=TrainerConfig(),
    model=model,
    x=x,
    y=y,
    val_x=valid_x,
    val_y=valid_y
)
import pdb; pdb.set_trace()
trainer.train()