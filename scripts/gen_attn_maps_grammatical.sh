CONFIG="./checkpoint/models/epoch139_head4/transformer_n7_config.json"
MODEL="./checkpoint/models/epoch139_head4/transformer_n7.pt"
OUTPUT_DIR="./data/outputs/heatmaps/gradient_probe"
NO_PLOT_FLAG="-np"

python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "((((((()))))))" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "(((((()())))))" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "((((()()()))))" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "(((()()()())))" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "((()()()()()))" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "(()()()()()())" $NO_PLOT_FLAG
python ./src/probe_model.py -c $CONFIG -m $MODEL -o $OUTPUT_DIR -s "()()()()()()()" $NO_PLOT_FLAG