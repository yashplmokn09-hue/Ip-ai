"""
IP-AIv1 Tokenizer — BPE with chat template support
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


PAD   = "<pad>"
UNK   = "<unk>"
BOS   = "<bos>"
EOS   = "<eos>"
SYS   = "<|system|>"
USR   = "<|user|>"
AST   = "<|assistant|>"
TOOL  = "<|tool|>"
TRESULT = "<|tool_result|>"

SPECIAL = [PAD, UNK, BOS, EOS, SYS, USR, AST, TOOL, TRESULT]


def train_tokenizer(text_files: list, vocab_size: int = 32000, save_path: str = "artifacts/tokenizer.json"):
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        min_frequency=2,
        show_progress=True,
    )
    tok.train(text_files, trainer)
    Path(save_path).parent.mkdir(exist_ok=True)
    tok.save(save_path)
    print(f"Tokenizer → {save_path}  vocab={tok.get_vocab_size()}")
    return tok


class IPAITokenizer:
    def __init__(self, path: str = "artifacts/tokenizer.json"):
        self._tok   = Tokenizer.from_file(path)
        v           = self._tok.get_vocab()
        self.pad_id = v[PAD]
        self.unk_id = v[UNK]
        self.bos_id = v[BOS]
        self.eos_id = v[EOS]
        self.sys_id = v[SYS]
        self.usr_id = v[USR]
        self.ast_id = v[AST]
        self.tool_id    = v[TOOL]
        self.tresult_id = v[TRESULT]

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id,
                self.sys_id, self.usr_id, self.ast_id,
                self.tool_id, self.tresult_id}
        return self._tok.decode([i for i in ids if i not in skip])

    def encode_chat(self, messages: list[dict], system: str = "") -> list[int]:
        """
        Encode a list of {"role": ..., "content": ...} messages.
        Roles: system | user | assistant | tool | tool_result
        """
        ids = [self.bos_id]
        if system:
            ids += [self.sys_id] + self.encode(system) + [self.eos_id]
        for m in messages:
            role = m["role"]
            if role == "user":
                ids += [self.usr_id] + self.encode(m["content"]) + [self.eos_id]
            elif role == "assistant":
                ids += [self.ast_id] + self.encode(m["content"]) + [self.eos_id]
            elif role == "tool":
                ids += [self.tool_id] + self.encode(m["content"]) + [self.eos_id]
            elif role == "tool_result":
                ids += [self.tresult_id] + self.encode(m["content"]) + [self.eos_id]
        ids += [self.ast_id]  # prime for assistant response
        return ids
