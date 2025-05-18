def is_valid_dyck_word(seq: str) -> bool:
    if len(seq) % 2 != 0:
        return False

    num_open_paren = 0

    for c in seq:
        if c == "(":
            num_open_paren += 1
        elif c == ")":
            num_open_paren -= 1
        else:
            print(f"Invalid Character: {c}")
            return False

        if num_open_paren < 0:
            return False
    return True