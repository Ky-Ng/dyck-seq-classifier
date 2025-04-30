def is_valid_parentheses(seq):
    balance = 0
    for char in seq:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        else:
            return False  # invalid character
        if balance < 0:
            return False  # more ')' than '(' at some point
    return balance == 0  # must be perfectly balanced