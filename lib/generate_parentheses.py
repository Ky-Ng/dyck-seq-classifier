import argparse
import os
# TODO: Understand this code fully
def generate_parentheses(n):
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)

    result = []
    backtrack('', 0, 0)
    return result

def main():
    parser = argparse.ArgumentParser(description="Generate all valid parentheses combinations.")
    parser.add_argument("-n", "--length", type=int, required=True, help="Number of pairs of parentheses")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory path")

    args = parser.parse_args()
    n = args.length
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    result = generate_parentheses(n)
    output_file = os.path.join(output_dir, f"parentheses_n{n}.txt")
    
    with open(output_file, "w") as f:
        for seq in result:
            f.write(seq + "\n")

    print(f"Generated {len(result)} sequences. Output written to {output_file}")

if __name__ == "__main__":
    main()
