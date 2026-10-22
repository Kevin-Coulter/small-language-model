"""Download WikiText-103 (raw), clean it, train the BPE tokenizer, and write
token ids to data/{train,val,test}.bin as uint16.

Usage: python prepare_data.py --vocab_size 8192
"""
import argparse
import os
import re
import time
import urllib.request
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from bpe import Tokenizer

BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/"
SPLITS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}

HEADING = re.compile(r"^((?:= )*=) (.*?) \1$")
QUOTES = re.compile(r'" (.*?) "')
CONTRACTION = re.compile(r" (n't|'t|'s|'re|'ve|'ll|'d|'m)\b")
SPACE_BEFORE = re.compile(r" ([.,;:!?%)\]}])")
SPACE_AFTER = re.compile(r"([(\[{$]) ")


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for files in SPLITS.values():
        for name in files:
            path = os.path.join(raw_dir, name)
            if os.path.exists(path):
                continue
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path + ".part")
            os.replace(path + ".part", path)


def load_lines(raw_dir, files):
    lines = []
    for name in files:
        lines.extend(pd.read_parquet(os.path.join(raw_dir, name), columns=["text"])["text"])
    return lines


def detokenize(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = QUOTES.sub(r'"\1"', s)
    s = CONTRACTION.sub(r"\1", s)
    s = SPACE_BEFORE.sub(r"\1", s)
    s = SPACE_AFTER.sub(r"\1", s)
    return s


def clean_line(line):
    """Return the cleaned line, or None if it is blank."""
    line = line.strip()
    if not line:
        return None
    m = HEADING.match(line)
    if m:
        marks = "=" * m.group(1).count("=")
        return f"{marks} {detokenize(m.group(2))} {marks}"
    return detokenize(line)


def is_title(line):
    s = line.strip()
    return s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")


def to_articles(lines):
    articles, cur = [], []
    for line in lines:
        if is_title(line) and cur:
            articles.append("\n".join(cur))
            cur = []
        c = clean_line(line)
        if c is not None:
            cur.append(c)
    if cur:
        articles.append("\n".join(cur))
    return articles


_tok = None


def _init_worker(merges):
    global _tok
    _tok = Tokenizer(merges)


def _encode_doc(doc):
    return np.array(_tok.encode(doc) + [_tok.eot_id], dtype=np.uint16)


def encode_split(docs, merges, workers):
    """Encode docs in parallel, each followed by EOT, as one uint16 array."""
    with Pool(workers, initializer=_init_worker, initargs=(merges,)) as pool:
        return np.concatenate(pool.map(_encode_doc, docs, chunksize=64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--min_word_freq", type=int, default=3,
                   help="ignore rarer pre-tokens when learning merges")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    t0 = time.time()
    raw_dir = os.path.join(args.data_dir, "raw")
    download(raw_dir)
    splits = {}
    for split, files in SPLITS.items():
        articles = to_articles(load_lines(raw_dir, files))
        text = "\n\n".join(articles) + "\n"
        with open(os.path.join(args.data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{split}: {len(articles):,} articles, {len(text) / 1e6:.1f}M chars")
        splits[split] = articles

    t1 = time.time()
    tok = Tokenizer()
    counts = Counter()
    for doc in splits["train"]:
        tok.count_words(doc, counts)
    counts = Counter({w: c for w, c in counts.items() if c >= args.min_word_freq})
    print(f"{len(counts):,} distinct pre-tokens")
    tok.train(counts, args.vocab_size)
    tok.save(os.path.join(args.data_dir, "tokenizer.json"))
    print(f"vocab size {tok.vocab_size}, trained in {time.time() - t1:.0f}s")

    for split, docs in splits.items():
        t1 = time.time()
        ids = encode_split(docs, tok.merges, args.workers)
        ids.tofile(os.path.join(args.data_dir, f"{split}.bin"))
        n_chars = sum(len(d) for d in docs)
        print(f"{split}: {len(ids):,} tokens, {n_chars / len(ids):.2f} chars/token, "
              f"encoded in {time.time() - t1:.0f}s")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
