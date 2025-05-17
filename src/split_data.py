import argparse
import random
import csv
import os

# The answer to the question of Life, the Universe, and Everything
random.seed(42)
OPT = {
    "TRAIN": 0.7,
    "VALID": 0.2,
    "TEST": 0.1
}


def read_in_file(path: str, class_label: str) -> list[str]:
    """
    Write an array of each line in the format "class_label, line_from_file"
    """
    ret = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            line = f"{class_label}, {line}"
            ret.append(line)
    return ret

def write_out_file(path: str, data: list[str]) -> None:
    """
    Writes data to `path` where `path` ends in .csv
    
    For example:
    "grammatical", "dyck_word"
    1,              (())
    """
    assert path.lower().endswith(".csv"), f"Path {path} must end with .csv"
    with open(path, "w") as f:
        # Write Header
        f.write("grammatical, dyck_word\n")

        # Write Rows
        for line in data:
            f.write(line + "\n")


def main():
    print("Splitting Data")

    parser = argparse.ArgumentParser()
    parser.add_argument("-fv", "--file_valid", required=True,
                        type=str, help="File name of valid parentheses")
    parser.add_argument("-fi", "--file_invalid", required=True,
                        type=str, help="File name of invalid parentheses")
    parser.add_argument("-o", "--output_dir", required=True, type=str,
                        help="Output Directory to write splits; Train-Dev-Test 70-20-10 Split; the `.csv` represents data with labels")
    parser.add_argument("-n", "--length", required=True, type=int,
                        help="Number of open parentheses, used for file naming")
    args = parser.parse_args()

    valid_src_path = args.file_valid
    invalid_src_path = args.file_invalid
    output_dir = args.output_dir
    n = args.length

    # Step 1) Read in Files
    grammatical_strs = read_in_file(valid_src_path, "1")
    ungrammatical_strs = read_in_file(invalid_src_path, "0")

    # Step 2) Randomize the Data
    random.shuffle(grammatical_strs)
    random.shuffle(ungrammatical_strs)

    # Step 3) Divide into Train-Dev-Test
    max_examples = min(len(grammatical_strs), len(ungrammatical_strs))
    num_train = int(OPT["TRAIN"] * max_examples)
    num_valid = int(OPT["VALID"] * max_examples)
    num_test = int(OPT["TEST"] * max_examples)

    train_range_start, train_range_end = 0, num_train
    valid_range_start, valid_range_end = train_range_end, train_range_end + num_valid
    test_range_start, test_range_end = valid_range_end, valid_range_end + num_test

    train_strs = (grammatical_strs[train_range_start:train_range_end] +
                  ungrammatical_strs[train_range_start:train_range_end])
    valid_strs = (grammatical_strs[valid_range_start:valid_range_end] +
                  ungrammatical_strs[valid_range_start:valid_range_end])
    test_strs = (grammatical_strs[test_range_start:test_range_end] +
                 ungrammatical_strs[test_range_start:test_range_end])

    # Step 4) Collate into csvs with labels
    write_out_file(os.path.join(output_dir, f"train_n{n}_unshuffled.csv"), train_strs)
    write_out_file(os.path.join(output_dir, f"valid_n{n}_unshuffled.csv"), valid_strs)
    write_out_file(os.path.join(output_dir, f"test_n{n}_unshuffled.csv"), test_strs)

    print(f"Out of {len(grammatical_strs)} valid strings and {len(ungrammatical_strs)} invalid strings, using {max_examples} examples from each class for a total of {max_examples*2}")
    print(
        f"""Splits are:
            Train Split: {OPT['TRAIN']} = {len(train_strs)}/{num_train*2}
            Valid Split: {OPT['VALID']} = {len(valid_strs)}/{num_valid*2}
            Test Split: {OPT['TEST']} = {len(test_strs)}/{num_test*2}
        """)


if __name__ == "__main__":
    main()
