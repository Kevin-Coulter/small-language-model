"""Pretrain the GPT model on data/*.bin with next-token prediction.

Usage:
    python train.py                 # full run, sized to fit --minutes
    python train.py --smoke         # ~1 minute check that everything works
    python train.py --resume        # continue from out/last.pt
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from bpe import Tokenizer
from model import GPT, GPTConfig

PROMPTS = ["The Beatles were", "= Pink Floyd =\n", "The city is located", "In 1998, the band"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--minutes", type=float, default=45, help="time budget")
    p.add_argument("--max_steps", type=int, default=0, help="overrides --minutes when > 0")
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=448)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.max_steps, args.eval_interval, args.log_interval, args.warmup = 100, 50, 10, 20
        args.out_dir += "_smoke"
        args.no_compile = True
    return args


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, ctx):
    """Mean loss over every non-overlapping block_size window of data."""
    was_training = model.training
    model.eval()
    n = (len(data) - 1) // block_size
    total = 0.0
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n)) * block_size
        x = torch.from_numpy(np.stack([data[i:i + block_size] for i in idx]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size] for i in idx]).astype(np.int64)).to(device)
        with ctx:
            _, loss = model(x, y)
        total += loss.item() * len(idx)
    model.train(was_training)
    return total / n


def main():
    args = parse_args()
    sys.stdout.reconfigure(errors="replace")
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        # T4 has no fast bfloat16.
        ctx = torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)
        use_scaler = not bf16
        print(f"device: cuda ({torch.cuda.get_device_name()}), {'bf16' if bf16 else 'fp16'} autocast")
    else:
        ctx = nullcontext()
        use_scaler = False
        print(f"device: {device}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tok = Tokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))
    train_data = np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16)
    print(f"train {len(train_data):,} tokens, val {len(val_data):,} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    raw_model = GPT(cfg).to(device)
    print(f"{raw_model.num_params() / 1e6:.2f}M parameters")
    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)

    step, best_val, max_steps = 0, float("inf"), args.max_steps
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume:
        ck = torch.load(last_path, map_location=device)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        max_steps = args.max_steps or ck["max_steps"]
        rng = np.random.default_rng(args.seed + step)
        print(f"resumed from step {step}")

    model = raw_model
    if device == "cuda" and not args.no_compile:
        model = torch.compile(raw_model)

    T, B = args.block_size, args.batch_size
    tokens_per_step = B * args.grad_accum * T
    placeholder = 10 ** 9

    def get_lr(it):
        total = max_steps or placeholder
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        frac = min(1.0, (it - args.warmup) / max(1, total - args.warmup))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))

    def get_batch():
        ix = rng.integers(0, len(train_data) - T - 1, B)
        x = torch.from_numpy(np.stack([train_data[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([train_data[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def train_step(m, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr
        total = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = get_batch()
            with ctx:
                _, loss = m(x, y)
            scaler.scale(loss / args.grad_accum).backward()
            total += loss.detach()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return total / args.grad_accum

    def sample(prompt):
        g = torch.Generator(device=device).manual_seed(args.seed)
        idx = torch.tensor([tok.encode(prompt)], device=device)
        out = list(raw_model.generate(idx, 64, eos_id=tok.eot_id, generator=g))
        return prompt + tok.decode(out)

    log_path = os.path.join(args.out_dir, "log.csv")
    if not args.resume:
        with open(log_path, "w") as f:
            f.write("step,tokens,train_loss,val_loss,val_ppl,lr,minutes\n")

    def log_row(step, train_loss="", val_loss="", val_ppl="", lr="", minutes=""):
        with open(log_path, "a") as f:
            f.write(f"{step},{step * tokens_per_step},{train_loss},{val_loss},{val_ppl},{lr},{minutes}\n")

    t_start = time.time()

    def run_eval(step):
        nonlocal best_val
        val = evaluate(raw_model, val_data, T, B, device, ctx)
        ppl = math.exp(val)
        minutes = (time.time() - t_start) / 60
        print(f"step {step}: val loss {val:.4f}, ppl {ppl:.1f}")
        log_row(step, val_loss=f"{val:.4f}", val_ppl=f"{ppl:.2f}", minutes=f"{minutes:.2f}")
        text = sample(PROMPTS[(step // args.eval_interval) % len(PROMPTS)])
        print(text)
        with open(os.path.join(args.out_dir, "samples.txt"), "a", encoding="utf-8") as f:
            f.write(f"--- step {step}, val loss {val:.4f}\n{text}\n\n")
        state = raw_model.state_dict()
        if val < best_val:
            best_val = val
            torch.save({"model": state, "config": asdict(cfg), "tokenizer": tok.to_dict(),
                        "step": step, "val_loss": val}, os.path.join(args.out_dir, "best.pt"))
        torch.save({"model": state, "config": asdict(cfg), "tokenizer": tok.to_dict(), "step": step,
                    "val_loss": val, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "best_val": best_val, "max_steps": max_steps}, last_path)

    if step == 0:
        run_eval(0)

    loss_sum, loss_n = torch.zeros((), device=device), 0
    t_log = time.time()
    t_bench = None
    first_step = True
    while max_steps == 0 or step < max_steps:
        lr = get_lr(step)
        try:
            loss = train_step(model, lr)
        except Exception as e:
            if model is raw_model or not first_step:
                raise
            print(f"torch.compile failed ({type(e).__name__}: {e}); continuing without it")
            model = raw_model
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(model, lr)
        first_step = False
        step += 1
        loss_sum += loss
        loss_n += 1

        if not max_steps:
            if step == 20:
                if device == "cuda":
                    torch.cuda.synchronize()
                t_bench = time.time()
            elif step == 50:
                if device == "cuda":
                    torch.cuda.synchronize()
                sec_per_step = (time.time() - t_bench) / 30
                remaining = args.minutes * 60 - (time.time() - t_start)
                max_steps = step + max(0, int(0.96 * remaining / sec_per_step))
                total_tokens = max_steps * tokens_per_step
                print(f"{sec_per_step * 1000:.0f} ms/step -> {max_steps} steps, "
                      f"{total_tokens / 1e6:.0f}M tokens, {total_tokens / len(train_data):.2f} epochs")

        if step % args.log_interval == 0:
            train_loss = (loss_sum / loss_n).item()
            dt = time.time() - t_log
            tps = loss_n * tokens_per_step / dt / 1e3
            minutes = (time.time() - t_start) / 60
            print(f"step {step}: loss {train_loss:.4f}, lr {lr:.2e}, {tps:.1f}k tok/s, {minutes:.1f} min")
            log_row(step, train_loss=f"{train_loss:.4f}", lr=f"{lr:.3e}", minutes=f"{minutes:.2f}")
            loss_sum, loss_n, t_log = torch.zeros((), device=device), 0, time.time()

        if step % args.eval_interval == 0 or step == max_steps:
            run_eval(step)

    best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    raw_model.load_state_dict(best["model"])
    test_data = np.fromfile(os.path.join(args.data_dir, "test.bin"), dtype=np.uint16)
    test = evaluate(raw_model, test_data, T, B, device, ctx)
    results = {
        "parameters": raw_model.num_params(),
        "steps": step,
        "tokens_seen": step * tokens_per_step,
        "best_step": best["step"],
        "val_loss": best["val_loss"],
        "val_ppl": math.exp(best["val_loss"]),
        "test_loss": test,
        "test_ppl": math.exp(test),
        "minutes": (time.time() - t_start) / 60,
        "config": asdict(cfg),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # About half the size, runs anywhere.
    demo = copy.copy(best)
    demo["model"] = {k: v.half() if v.is_floating_point() else v for k, v in best["model"].items()}
    torch.save(demo, os.path.join(args.out_dir, "demo.pt"))


if __name__ == "__main__":
    main()
