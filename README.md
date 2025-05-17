# dyck-gpt
Modelling dependency in LLMs using the Dyck Language

## Data Generation
```
python src/data_gen.py -o data/input -n 12
```
## Splitting Data
```
python src/split_data.py -fv data/input/valid_parentheses_n12.txt -fi data/input/invalid_parentheses_n12.txt -n 12 -o ./data/splits
```
## Training
```
python src/train_loop.py -n 7 -e 160
```

## Probing the Model
```
python ./src/probe_model.py --config_path=./checkpoint/models/epoch156/transformer_n14_config.json --model_checkpoint=./checkpoint/models/epoch156/transformer_n14.pt
```