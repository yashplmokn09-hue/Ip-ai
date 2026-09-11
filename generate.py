"""
IP-AIv1 Generation — supports streaming and standard generation
"""

import torch
from typing import Iterator
from model import IPAIv1
from tokenizer import IPAITokenizer


@torch.no_grad()
def generate(
    model          : IPAIv1,
    tokenizer      : IPAITokenizer,
    messages       : list[dict],
    system         : str   = "You are IP-AIv1, a helpful AI assistant.",
    max_new_tokens : int   = 512,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 0.95,
    rep_penalty    : float = 1.15,
) -> str:
    return "".join(stream(model, tokenizer, messages, system,
                          max_new_tokens, temperature, top_k, top_p, rep_penalty))


@torch.no_grad()
def stream(
    model          : IPAIv1,
    tokenizer      : IPAITokenizer,
    messages       : list[dict],
    system         : str   = "You are IP-AIv1, a helpful AI assistant.",
    max_new_tokens : int   = 512,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 0.95,
    rep_penalty    : float = 1.15,
) -> Iterator[str]:
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode_chat(messages, system=system)
    x   = torch.tensor([ids[-model.block_size:]], dtype=torch.long, device=device)
    gen = []

    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Repetition penalty
        if rep_penalty != 1.0 and gen:
            for tid in set(gen):
                logits[0, tid] /= rep_penalty

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p nucleus
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

        probs  = torch.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1).item()

        if nxt == tokenizer.eos_id:
            break

        gen.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)[:, -model.block_size:]
        yield tokenizer.decode([nxt])


# Import F for top-p
import torch.nn.functional as F
