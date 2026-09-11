"""
IP-AIv1 API — Maximum Level
============================
- Groq/OpenAI-compatible /v1/chat/completions
- Streaming (SSE)
- Real-time web search + Wikipedia tools
- Raw /v1/generate endpoint
- Health + model info endpoints
"""

import time, uuid, json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model import IPAIv1
from tokenizer import IPAITokenizer
from generate import generate as model_generate, stream as model_stream
from dispatcher import maybe_run_tool, SYSTEM_PROMPT

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS = Path("artifacts")
MODEL_ID  = "ip-ai-v1"

MODEL    : Optional[IPAIv1]        = None
TOKENIZER: Optional[IPAITokenizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER
    tok_path   = ARTIFACTS / "tokenizer.json"
    model_path = ARTIFACTS / "model_best.pt"
    if tok_path.exists() and model_path.exists():
        TOKENIZER = IPAITokenizer(str(tok_path))
        ckpt      = torch.load(str(model_path), map_location=DEVICE)
        cfg       = ckpt["cfg"]
        MODEL     = IPAIv1(
            vocab_size = TOKENIZER.vocab_size,
            block_size = cfg["block_size"],
            d_model    = cfg["d_model"],
            n_layers   = cfg["n_layers"],
            n_heads    = cfg["n_heads"],
            n_kv_heads = cfg["n_kv_heads"],
            dropout    = 0.0,
        ).to(DEVICE)
        MODEL.load_state_dict(ckpt["model"])
        MODEL.eval()
        print(f"IP-AIv1 ready | params={MODEL.num_parameters():,} | device={DEVICE}")
    else:
        print("WARNING: No model artifacts found. Run train_colab.py first.")
    yield


app = FastAPI(title="IP-AIv1 API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"])


# ── Schemas ────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role   : str
    content: str


class ChatRequest(BaseModel):
    model        : str         = MODEL_ID
    messages     : list[Message]
    max_tokens   : int         = Field(512, ge=1, le=2048)
    temperature  : float       = Field(0.8, ge=0.0, le=2.0)
    top_p        : float       = Field(0.95, ge=0.0, le=1.0)
    top_k        : int         = Field(50, ge=1, le=200)
    stream       : bool        = False
    use_tools    : bool        = True   # enable web search


class GenerateRequest(BaseModel):
    prompt       : str
    max_new_tokens: int  = 256
    temperature  : float = 0.8
    top_k        : int   = 50
    top_p        : float = 0.95


# ── Helper ─────────────────────────────────────────────────────────────────
def _require_model():
    if not MODEL or not TOKENIZER:
        raise HTTPException(503, "Model not loaded. Run train_colab.py first.")


def _to_dicts(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _run_with_tools(messages: list[dict], req: ChatRequest, max_tool_rounds: int = 3) -> str:
    """Generate a response, executing tool calls if the model produces them."""
    conv = list(messages)
    for _ in range(max_tool_rounds):
        reply = model_generate(
            MODEL, TOKENIZER, conv,
            system         = SYSTEM_PROMPT if req.use_tools else "You are IP-AIv1, a helpful AI assistant.",
            max_new_tokens = req.max_tokens,
            temperature    = req.temperature,
            top_k          = req.top_k,
            top_p          = req.top_p,
        )
        result, did_call = maybe_run_tool(reply)
        if not did_call:
            return reply
        # Feed tool result back
        conv.append({"role": "assistant",   "content": reply})
        conv.append({"role": "tool_result", "content": result})
    return reply


async def _sse_stream(messages: list[dict], req: ChatRequest) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI/Groq format."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    for token in model_stream(
        MODEL, TOKENIZER, messages,
        system         = SYSTEM_PROMPT if req.use_tools else "You are IP-AIv1, a helpful AI assistant.",
        max_new_tokens = req.max_tokens,
        temperature    = req.temperature,
        top_k          = req.top_k,
        top_p          = req.top_p,
    ):
        chunk = {
            "id"     : cid,
            "object" : "chat.completion.chunk",
            "created": int(time.time()),
            "model"  : MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    # Final chunk
    done = {
        "id": cid, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "model"     : MODEL_ID,
        "status"    : "ready" if MODEL else "no_model",
        "device"    : DEVICE,
        "parameters": MODEL.num_parameters() if MODEL else 0,
        "tools"     : ["web_search", "wikipedia_search", "fetch_page"],
        "version"   : "1.0.0",
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data"  : [{
            "id"        : MODEL_ID,
            "object"    : "model",
            "created"   : 1700000000,
            "owned_by"  : "ip-ai",
        }],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    _require_model()
    msgs = _to_dicts(req.messages)

    if req.stream:
        return StreamingResponse(
            _sse_stream(msgs, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    reply = _run_with_tools(msgs, req)
    return {
        "id"     : f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object" : "chat.completion",
        "created": int(time.time()),
        "model"  : MODEL_ID,
        "choices": [{
            "index"       : 0,
            "message"     : {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens"    : sum(len(TOKENIZER.encode(m["content"])) for m in msgs),
            "completion_tokens": len(TOKENIZER.encode(reply)),
        },
    }


@app.post("/v1/generate")
def raw_generate(req: GenerateRequest):
    _require_model()
    out = model_generate(
        MODEL, TOKENIZER,
        [{"role": "user", "content": req.prompt}],
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return {"prompt": req.prompt, "output": out, "model": MODEL_ID}


@app.get("/v1/tools")
def list_tools():
    from web_tools import TOOL_SCHEMAS
    return {"tools": TOOL_SCHEMAS}


@app.post("/v1/tools/search")
def tool_search(body: dict):
    """Direct web search endpoint — no model needed."""
    from web_tools import web_search
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "query required")
    return {"results": web_search(query, max_results=body.get("max_results", 5))}


@app.post("/v1/tools/wikipedia")
def tool_wiki(body: dict):
    """Direct Wikipedia lookup — no model needed."""
    from web_tools import wikipedia_search
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "query required")
    return wikipedia_search(query, sentences=body.get("sentences", 5))
