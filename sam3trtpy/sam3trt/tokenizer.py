import json


class ClipTokenizer:
    _PADDING_TOKEN = 49407

    def __init__(self, merges_path, vocab_path):
        self._merge_rank = self._load_merges(merges_path)
        with open(vocab_path) as f:
            self._vocab = json.load(f)

    @staticmethod
    def _load_merges(path):
        with open(path) as f:
            lines = [line.split() for line in f if line.strip()][1:]
        return {(pair[0], pair[1]): rank for rank, pair in enumerate(p for p in lines if len(p) >= 2)}

    def _bpe(self, word):
        symbols = list(word[:-1]) + [word[-1] + "</w>"]
        while len(symbols) > 1:
            ranks = [self._merge_rank.get(pair) for pair in zip(symbols, symbols[1:])]
            candidates = [(rank, i) for i, rank in enumerate(ranks) if rank is not None]
            if not candidates:
                break
            _, i = min(candidates)
            symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        return [s for s in symbols if s in self._vocab]

    def tokenize(self, text):
        tokens = ["<|startoftext|>"]
        for word in text.lower().split():
            tokens += self._bpe(word)
        tokens.append("<|endoftext|>")
        return [self._vocab.get(token, self._PADDING_TOKEN) for token in tokens]
