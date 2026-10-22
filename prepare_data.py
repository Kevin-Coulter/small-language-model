"""Download WikiText-103 (raw), undo its tokenization, and write plain-text splits."""
import argparse
import os
import re
import time
import urllib.request

import pandas as pd

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()

    t0 = time.time()
    raw_dir = os.path.join(args.data_dir, "raw")
    download(raw_dir)
    for split, files in SPLITS.items():
        articles = to_articles(load_lines(raw_dir, files))
        text = "\n\n".join(articles) + "\n"
        with open(os.path.join(args.data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{split}: {len(articles):,} articles, {len(text) / 1e6:.1f}M chars")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
