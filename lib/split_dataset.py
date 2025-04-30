import argparse
import os
import random

def split_dataset(corpus, seed=42):
    random.seed(seed)
    random.shuffle(corpus)
    
    total = len(corpus)
    train_size = total // 2
    remaining = total - train_size
    val_size = int(0.6 * remaining)  # 60% of remaining = 30% of total
    test_size = remaining - val_size  # 40% of remaining = 20% of total
    
    train_set = corpus[:train_size]
    val_set = corpus[train_size:train_size + val_size]
    test_set = corpus[train_size + val_size:]
    
    return train_set, val_set, test_set

def write_split(split, filename):
    with open(filename, "w") as f:
        for line in split:
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Split valid parentheses data into train/val/test sets.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input .txt file with one sequence per line")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for the split files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input, "r") as f:
        corpus = [line.strip() for line in f if line.strip()]

    train_set, val_set, test_set = split_dataset(corpus)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    write_split(train_set, os.path.join(args.output, f"{basename}_train.txt"))
    write_split(val_set, os.path.join(args.output, f"{basename}_val.txt"))
    write_split(test_set, os.path.join(args.output, f"{basename}_test.txt"))

    print(f"Written {len(train_set)} train, {len(val_set)} val, and {len(test_set)} test samples to '{args.output}'")

if __name__ == "__main__":
    main()
