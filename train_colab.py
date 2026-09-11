"""
IP-AIv1 — Maximum Training Script for Google Colab T4 GPU
=========================================================
Phase 1: Pretrain on OpenWebText (language modeling)
Phase 2: Instruction fine-tune on Alpaca + FLAN data
Phase 3: Upload to Hugging Face Hub

Runtime: ~3-4 hours on Colab T4
"""

# ── 0. Install ─────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "datasets", "tokenizers", "huggingface_hub", "tqdm"], check=True)

# ── 1. Imports ─────────────────────────────────────────────────────────────
import os, math, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm

from ip_ai.model import IPAIv1
from ip_ai.tokenizer import train_tokenizer, IPAITokenizer, USR, AST

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 2. Config ──────────────────────────────────────────────────────────────
CFG = {
    # ---- Model (50M params) ----
    "vocab_size"  : 32000,
    "block_size"  : 1024,
    "d_model"     : 768,
    "n_layers"    : 12,
    "n_heads"     : 12,
    "n_kv_heads"  : 4,
    "dropout"     : 0.1,

    # ---- Pretrain ----
    "pretrain_samples" : 100_000,
    "pretrain_steps"   : 10_000,
    "pretrain_batch"   : 8,
    "pretrain_grad_acc": 8,       # effective batch = 64
    "pretrain_lr"      : 3e-4,

    # ---- Fine-tune ----
    "finetune_steps"   : 2_000,
    "finetune_batch"   : 4,
    "finetune_grad_acc": 4,
    "finetune_lr"      : 1e-4,

    # ---- Common ----
    "warmup_steps" : 400,
    "eval_every"   : 500,
    "save_every"   : 1000,

    # ---- Hugging Face ----
    "hf_token"    : "",                       # ← paste your HF write token
    "hf_repo"     : "your-username/IP-AIv1",  # ← change to your HF username
}


# ── 3. Download & prepare data ─────────────────────────────────────────────
print("\n[1/6] Downloading pretraining data (OpenWebText)...")
owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
texts = []
for i, s in enumerate(owt):
    texts.append(s["text"])
    if i >= CFG["pretrain_samples"]: break
print(f"  {len(texts):,} documents")

raw_path = ARTIFACTS / "raw_pretrain.txt"
with open(raw_path, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t.replace("\n", " ") + "\n")


# ── 4. Train tokenizer ─────────────────────────────────────────────────────
print("\n[2/6] Training BPE tokenizer (vocab=32k)...")
tok_path = str(ARTIFACTS / "tokenizer.json")
train_tokenizer([str(raw_path)], vocab_size=CFG["vocab_size"], save_path=tok_path)
tokenizer = IPAITokenizer(tok_path)
CFG["vocab_size"] = tokenizer.vocab_size   # actual size after training
print(f"  Actual vocab: {tokenizer.vocab_size}")


# ── 5. Pretrain dataset ────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        all_ids = []
        for t in tqdm(texts, desc="Encoding pretrain"):
            all_ids.extend(tokenizer.encode(t) + [tokenizer.eos_id])
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.block = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block - 1)

    def __getitem__(self, i):
        return self.data[i:i+self.block], self.data[i+1:i+self.block+1]


# ── 6. Build model ─────────────────────────────────────────────────────────
print("\n[3/6] Building IP-AIv1 (~50M params)...")
model = IPAIv1(
    vocab_size = CFG["vocab_size"],
    block_size = CFG["block_size"],
    d_model    = CFG["d_model"],
    n_layers   = CFG["n_layers"],
    n_heads    = CFG["n_heads"],
    n_kv_heads = CFG["n_kv_heads"],
    dropout    = CFG["dropout"],
).to(DEVICE)
print(f"  Parameters: {model.num_parameters():,}")


def make_optimizer(model, lr):
    decay     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay  = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / warmup
    t = (step - warmup) / (total - warmup)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * t)))


def train_loop(model, loader, optimizer, scheduler, scaler, steps, grad_acc, desc):
    model.train()
    it       = iter(loader)
    best     = float("inf")
    log      = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0

        for _ in range(grad_acc):
            try:    x, y = next(it)
            except: it = iter(loader); x, y = next(it)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                _, loss = model(x, y)
                loss = loss / grad_acc
            scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  [{desc}] step={step:>5} loss={loss_acc:.4f} lr={lr:.2e}")
            log.append({"step": step, "loss": round(loss_acc, 4)})

        if step % CFG["save_every"] == 0:
            ckpt = {"step": step, "model": model.state_dict(), "cfg": CFG}
            torch.save(ckpt, ARTIFACTS / f"ckpt_{desc}_{step}.pt")
            if loss_acc < best:
                best = loss_acc
                torch.save(ckpt, ARTIFACTS / "model_best.pt")

    return log


# ── 7. Phase 1 — Pretrain ──────────────────────────────────────────────────
print("\n[4/6] Phase 1 — Pretraining...")
pretrain_ds     = PretrainDataset(texts, tokenizer, CFG["block_size"])
pretrain_loader = DataLoader(pretrain_ds, batch_size=CFG["pretrain_batch"],
                             shuffle=True, num_workers=2, pin_memory=True)

optimizer = make_optimizer(model, CFG["pretrain_lr"])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: lr_lambda(s, CFG["warmup_steps"], CFG["pretrain_steps"]))
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

pretrain_log = train_loop(model, pretrain_loader, optimizer, scheduler, scaler,
                          CFG["pretrain_steps"], CFG["pretrain_grad_acc"], "pretrain")
torch.save({"model": model.state_dict(), "cfg": CFG}, ARTIFACTS / "model_pretrain.pt")
print("  Pretrain complete.")


# ── 8. Phase 2 — Instruction Fine-tuning ───────────────────────────────────
print("\n[5/6] Phase 2 — Instruction fine-tuning...")


class InstructDataset(Dataset):
    def __init__(self, tokenizer, block_size, max_samples=10000):
        self.examples = []
        # Load Alpaca instruction data
        try:
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            for row in tqdm(ds, desc="Alpaca", total=min(max_samples, len(ds))):
                if len(self.examples) >= max_samples: break
                inp  = row.get("input", "").strip()
                inst = row["instruction"].strip()
                out  = row["output"].strip()
                prompt = inst + ("\n" + inp if inp else "")
                messages = [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": out},
                ]
                ids = tokenizer.encode_chat(messages)
                if len(ids) <= block_size:
                    self.examples.append(torch.tensor(ids, dtype=torch.long))
        except Exception as e:
            print(f"  Warning: {e}")

        print(f"  Instruction examples: {len(self.examples)}")
        self.block = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids = self.examples[i]
        pad = self.block - len(ids)
        if pad > 0:
            ids = F.pad(ids, (0, pad), value=0)
        else:
            ids = ids[:self.block]
        x = ids[:-1]
        y = ids[1:].clone()
        return x, y


instruct_ds     = InstructDataset(tokenizer, CFG["block_size"])
instruct_loader = DataLoader(instruct_ds, batch_size=CFG["finetune_batch"],
                             shuffle=True, num_workers=2, pin_memory=True)

optimizer = make_optimizer(model, CFG["finetune_lr"])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: lr_lambda(s, 100, CFG["finetune_steps"]))

finetune_log = train_loop(model, instruct_loader, optimizer, scheduler, scaler,
                          CFG["finetune_steps"], CFG["finetune_grad_acc"], "finetune")

torch.save({"model": model.state_dict(), "cfg": CFG}, ARTIFACTS / "model_final.pt")
# Also save as best if better
print("  Fine-tuning complete.")

# Save logs
with open(ARTIFACTS / "train_log.json", "w") as f:
    json.dump({"pretrain": pretrain_log, "finetune": finetune_log}, f, indent=2)


# ── 9. Upload to Hugging Face ──────────────────────────────────────────────
print("\n[6/6] Uploading to Hugging Face Hub...")
if CFG["hf_token"] and "your-username" not in CFG["hf_repo"]:
    login(token=CFG["hf_token"])
    api = HfApi()
    api.create_repo(CFG["hf_repo"], exist_ok=True, repo_type="model")

    # Write model card
    card = f"""---
language: en
license: apache-2.0
tags:
  - ip-ai
  - causal-lm
  - text-generation
---

# IP-AIv1

Original causal language model with ~50M parameters.
Architecture: RoPE + GQA + SwiGLU + RMSNorm.

## Usage
```python
from huggingface_hub import hf_hub_download
import torch
from ip_ai.model import IPAIv1
from ip_ai.tokenizer import IPAITokenizer
from ip_ai.generate import generate

ckpt = torch.load(hf_hub_download("{CFG['hf_repo']}", "model_final.pt"))
tok  = IPAITokenizer(hf_hub_download("{CFG['hf_repo']}", "tokenizer.json"))
model = IPAIv1(**{{k: ckpt['cfg'][k] for k in
    ['vocab_size','block_size','d_model','n_layers','n_heads','n_kv_heads','dropout']}})
model.load_state_dict(ckpt['model'])
print(generate(model, tok, [{{'role':'user','content':'Hello!'}}]))
```
"""
    (ARTIFACTS / "README.md").write_text(card)

    api.upload_folder(
        folder_path=str(ARTIFACTS),
        repo_id=CFG["hf_repo"],
        repo_type="model",
    )
    print(f"  Published → https://huggingface.co/{CFG['hf_repo']}")
else:
    print("  Skipped — fill hf_token and hf_repo in CFG to publish.")

print("\nAll done.")
