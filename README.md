# Pretraining a Small Language Model from Scratch

This is CMPSC 448 Homework 3. It trains a GPT-style decoder-only transformer from random initialization on WikiText-103 with next-token prediction. Everything is written in plain PyTorch, including the tokenizer.

## Dataset

[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-103-raw-v1`) is about 100M words taken from Good and Featured articles on Wikipedia ([Merity et al., 2016, arXiv:1609.07843](https://arxiv.org/abs/1609.07843)). It is licensed under CC BY-SA 3.0 and the GFDL.

The raw text comes pre-tokenized, so `prepare_data.py` undoes that before training:

- Number separators are rejoined: `guest @-@ starring` -> `guest-starring`, `1 @,@ 500 @.@ 5` -> `1,500.5`.
- Spaces around punctuation are removed: `actor .` -> `actor.`, `( about 3 % )` -> `(about 3%)`, `$ 1` -> `$1`, `didn 't` -> `didn't`.
- Quotes are paired within a line: `" Scott Parry "` -> `"Scott Parry"`.
- Headings are kept in wiki form: `= = Career = =` -> `== Career ==`.
- A level-1 heading starts a new article, and articles are separated by an end-of-text token.
- The official train, validation and test splits are used as-is.

## Tokenizer

`bpe.py` is a byte-level BPE tokenizer written from scratch with a vocabulary of 8,192 tokens: the 256 byte values, 7,935 learned merges, and `<|endoftext|>`. Text is first split with a GPT-2-style regex into words, runs of up to three digits, punctuation and whitespace. Each piece starts as its UTF-8 bytes, and training repeatedly fuses the most frequent adjacent pair. Since every byte is in the base vocabulary, any string can be encoded, including characters never seen in training.

## Model

`model.py` follows the decoder-only architecture from lecture. The token and positional embeddings are added together, passed through a stack of blocks that each apply masked multi-head self-attention and a feed-forward network on the residual stream, and then a final norm and a linear layer produce a distribution over the next token.

| | |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Feed-forward size | 1536 (GELU) |
| Context length | 512 |
| Vocabulary | 8,192 |
| Parameters | 14.0M |

It uses pre-norm LayerNorm, learned positional embeddings, input and output embeddings tied together, and residual projections initialized with standard deviation 0.02/sqrt(2 * layers).

## Training

- Objective: maximize P(w_1, ..., w_n) = prod_i P(w_i | w_<i), i.e. minimize the cross-entropy of each next token.
- AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices only, and gradient clipping at 1.0.
- Learning rate: linear warmup over 200 steps to 1e-3, then cosine decay to 1e-4.
- Batches of 64 sequences x 512 tokens = 32,768 tokens per step.
- fp16 autocast with loss scaling on a T4, compiled with `torch.compile`.
- No dropout, since each token is seen roughly once and there is little to overfit.
- Seed 448.

`train.py` takes a time budget with `--minutes`: after 50 steps it measures how fast training runs and sets the number of steps and the LR schedule to fit. Every 250 steps it computes the loss over the full validation split, writes a generated sample to `samples.txt`, and saves the checkpoint with the best validation loss, which is what the test set is evaluated on at the end.

## Results

| | |
|---|---|
| Training time | |
| Tokens seen | |
| Validation loss / perplexity | |
| Test loss / perplexity | |

The perplexity is per BPE token, so it isn't directly comparable to the word-level WikiText-103 numbers reported in papers.

![Training and validation loss](loss.png)

## Running it

On Colab, open `colab.ipynb` with a T4 runtime and run the cells in order. Locally:

```
pip install -r requirements.txt
python prepare_data.py
python train.py --smoke
python train.py --minutes 45
python plot_loss.py
python generate.py --ckpt out/demo.pt
```

`generate.py` runs fine on a laptop CPU (use `--device mps` on Apple Silicon). Without `--prompt` it is interactive. It samples with temperature 0.8, top-k 50, top-p 0.95 and repetition penalty 1.1, and uses a KV cache so each new token costs one forward step. Some prompts to try:

- `The Beatles were`
- `= Pink Floyd =\n`
- `In 1998, the band released`
- `The city is located`
- `The game was released`

## Files

| File | Description |
|---|---|
| `prepare_data.py` | download, clean, tokenize |
| `bpe.py` | byte-level BPE tokenizer |
| `model.py` | GPT model and sampling |
| `train.py` | pretraining loop |
| `generate.py` | prompt a checkpoint |
| `plot_loss.py` | loss curve |
| `colab.ipynb` | full pipeline on Colab |
| `requirements.txt` | dependencies |
