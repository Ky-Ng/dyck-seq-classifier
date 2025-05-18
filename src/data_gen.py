import argparse
import math
import os

from dyck_lib import is_valid_dyck_word


def generate_dyck_words(n: int) -> list[str]:
    """
    Generates all valid dyck words with n open and n close parentheses
    """
    def backtrack(base: str, num_open: int, num_close: int) -> None:
        # Case 1) Completed a dyck word
        if len(base) == 2*n:
            ret.append(base)
            return

        # Case 2) Add an open parentheses i.e. ( => ((
        if num_open < n:
            backtrack(base + "(", num_open + 1, num_close)

        # Case 3) Complete a parentheses i.e. (( => (()
        if num_open > num_close:
            backtrack(base + ")", num_open, num_close + 1)

    ret = []
    backtrack(base="", num_open=0, num_close=0)
    return ret


def generate_invalid_dyck_words(n: int) -> list[str]:
    """
    Generates all sequences with n open and n close parentheses
    such that word_1 = "(" and word_{2n} = ")", where word_i = ith position of a string

    Returns:
        invalid_dyck_words

    Algorithm:
        1. Generate all sequences with n-1 open and close parentheses (may or may not be balanced)
        2. Prepend "(" and append ")"
        3. Remove valid dyck_word
    """
    
    # fix the first and last parentheses to be open

    def backtrack(seq: str, num_open: int, num_close: int) -> None:
        """
        Generate sequences of length 2n with n open and n close parentheses
        """
        if len(seq) == 2*(n-1):
            combinations.append("(" + seq + ")")
        if num_open < n-1:
            backtrack(seq + "(", num_open + 1, num_close)
        if num_close < n-1:
            backtrack(seq + ")", num_open, num_close + 1)
    
    # Populate combinations with all combos of n "(" and n ")"
    combinations = []
    backtrack("", num_open=0, num_close=0)
    
    # return only the invalid ones
    invalid_dyck_words = [s for s in combinations if not is_valid_dyck_word(s)]
    return invalid_dyck_words

def catalan(n):
    return math.comb(2*n, n) // (n+1)

def main():
    print("Initializing Data Generation of Dyck Words")

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--length", type=int,
                        required=True, help="Number of Open Parentheses")
    parser.add_argument("-o", "--output_dir", type=str,
                        required=False, help="Output directory to write results")

    args = parser.parse_args()
    n = args.length
    dir = args.output_dir

    valid_dyck_words = generate_dyck_words(n)
    invalid_dyck_words = generate_invalid_dyck_words(n)

    # Write out Dyck words
    if dir:
        # Create output directories if needed
        os.makedirs(dir, exist_ok=True)

        with open(os.path.join(dir, f"valid_parentheses_n{n}.txt"), mode='w') as f:
            for word in valid_dyck_words:
                f.write(word + "\n")
        
        with open(os.path.join(dir, f"invalid_parentheses_n{n}.txt"), mode='w') as f:
            for word in invalid_dyck_words:
                f.write(word + "\n")

        print(f"Writing to {dir}")

    print(f"Generated {len(valid_dyck_words)} out of {catalan(n)} combinations; Generated {len(invalid_dyck_words)} invalid words")


if __name__ == "__main__":
    main()
