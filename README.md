# dyck-gpt
Modelling dependency in LLMs using the Dyck Language

## Data Generation
```
src/data_gen.py -o data/input -n 12
```
## Splitting Data
```
python src/split_data.py -fv data/input/valid_parentheses_n12.txt -fi data/input/invalid_parentheses_n12.txt -n 12 -o ./data/splits
```