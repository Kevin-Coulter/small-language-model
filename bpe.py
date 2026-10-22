"""Byte-level BPE tokenizer.

Text is split into pre-tokens with a regex, and each pre-token starts as its
UTF-8 bytes. Merges are learned by repeatedly fusing the most frequent adjacent
pair into a new token. Every byte is in the base vocab, so any string can be
encoded.
"""
import heapq
import json
import re
from collections import Counter, defaultdict

# [^\W\d_] is a letter, and digits are grouped in runs of up to 3.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


def _merge(word, pair, new_id):
    a, b = pair
    out, i, n = [], 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges=None):
        self.pattern = re.compile(SPLIT_PATTERN)
        self.merges = [tuple(m) for m in merges or []]
        self._build()

    def _build(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.vocab_size = self.eot_id + 1
        self._cache = {}

    def train(self, word_counts, vocab_size, verbose=True):
        """Learn merges from a Counter of pre-token -> frequency."""
        n_merges = vocab_size - 257  # last id is reserved for EOT
        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())

        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for idx, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(idx)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg:
                continue
            if -neg < 2:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for idx in where.pop(pair, ()):
                w, f = words[idx], freqs[idx]
                nw = _merge(w, pair, new_id)
                if len(nw) == len(w):
                    continue
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += f
                    where[p].add(idx)
                    changed.add(p)
                words[idx] = nw
            for p in changed:
                c = pair_counts[p]
                if c <= 0:
                    del pair_counts[p]
                else:
                    heapq.heappush(heap, (-c, p))
            if verbose and len(merges) % 1000 == 0:
                print(f"merge {len(merges)}/{n_merges}: count {-neg}")

        self.merges = merges
        self._build()

    def count_words(self, text, counter=None):
        counter = Counter() if counter is None else counter
        counter.update(self.pattern.findall(text))
        return counter

    def _encode_word(self, word):
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for word in self.pattern.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        eot = EOT.encode("utf-8")
        return b"".join(eot if i == self.eot_id else self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self):
        return {"pattern": SPLIT_PATTERN, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
