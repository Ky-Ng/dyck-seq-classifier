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
python ./src/probe_model.py -c ./checkpoint/models/epoch139_head4/transformer_n7_config.json -m ./checkpoint/models/epoch139_head4/transformer_n7.pt -o ./data/outputs/heatmaps -s "()()()()()()()"
```

### Generate Heatmaps Automatically
```
chmod +x ./scripts/gen_attn_maps.sh
./scripts/gen_attn_maps.sh
```