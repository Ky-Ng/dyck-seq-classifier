class Tokenizer():
    def __init__(self, symbols: list[str]):
        self.token_mapping = dict()
        self.id_to_token_mapping = dict()

        for idx, symbol in enumerate(symbols):
            self.token_mapping[symbol] = idx
            self.id_to_token_mapping[idx] = symbol

    def encode(self, seq: str) -> list[int]:
        result = []
        for c in seq:
            result.append(self.token_mapping[c])

        return result

    def decode(self, tokens: list[int]) -> str:
        result = []
        for id in tokens:
            result.append(self.id_to_token_mapping[id])
        return "".join(result)