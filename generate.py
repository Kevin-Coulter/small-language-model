"""Prompt a trained checkpoint and stream the continuation.

Usage:
    python generate.py --ckpt out/demo.pt                  # interactive
    python generate.py --ckpt out/demo.pt --prompt "The Beatles were"
"""
import argparse
import sys

import torch

from bpe import Tokenizer
from model import GPT, GPTConfig


def load(path, device):
    ck = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ck["config"]))
    # demo.pt is fp16, and fp16 matmuls are slow or unsupported on CPU.
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.to(device).eval()
    return model, Tokenizer.from_dict(ck["tokenizer"]), ck


def stream(model, tok, prompt, device, opts, seed=None):
    """Print the continuation of prompt as it is generated and return it."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    ids = tok.encode(prompt) or [tok.eot_id]
    idx = torch.tensor([ids], device=device)
    out, shown = [], ""
    for t in model.generate(idx, opts.max_new_tokens, temperature=opts.temperature, top_k=opts.top_k,
                            top_p=opts.top_p, repetition_penalty=opts.repetition_penalty,
                            eos_id=tok.eot_id, generator=g):
        if t == tok.eot_id:
            break
        out.append(t)
        text = tok.decode(out)
        if text.endswith("�"):  # incomplete multi-byte character
            continue
        print(text[len(shown):], end="", flush=True)
        shown = text
    print()
    return shown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/demo.pt")
    p.add_argument("--prompt")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda, cpu, or mps")
    args = p.parse_args()
    sys.stdout.reconfigure(errors="replace")

    model, tok, ck = load(args.ckpt, args.device)
    print(f"{args.ckpt}: {model.num_params() / 1e6:.1f}M parameters, "
          f"step {ck['step']}, val loss {ck['val_loss']:.3f}")

    if args.prompt is not None:
        prompt = args.prompt.replace("\\n", "\n")
        for i in range(args.num_samples):
            print(prompt, end="")
            stream(model, tok, prompt, args.device, args,
                   None if args.seed is None else args.seed + i)
        return

    print("Type a prompt and press enter (\\n for a newline, empty line or Ctrl-C to quit).")
    while True:
        try:
            line = input("\n> ")
        except EOFError:
            break
        if not line:
            break
        prompt = line.replace("\\n", "\n")
        print(prompt, end="")
        stream(model, tok, prompt, args.device, args, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
