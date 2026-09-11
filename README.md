# IP-AIv1

Your own original AI model — built from scratch.

**Architecture:** RoPE + Grouped Query Attention + SwiGLU + RMSNorm  
**Size:** ~50M parameters  
**Internet:** Real-time web search + Wikipedia (no API key needed)  
**API:** Groq/OpenAI-compatible REST  
**UI:** Streaming chat with web search toggle  

---

## Project Structure

```
IP-AIv1/
├── ip_ai/
│   ├── model.py        # Transformer (RoPE, GQA, SwiGLU, RMSNorm)
│   ├── tokenizer.py    # BPE tokenizer with chat template
│   └── generate.py     # Streaming + standard generation
├── tools/
│   ├── web_tools.py    # DuckDuckGo + Wikipedia + page fetch
│   └── dispatcher.py   # Tool call parser and executor
├── train_colab.py      # Full training script (Colab T4 GPU)
├── api.py              # FastAPI — Groq-compatible + streaming
├── index.html          # Chat UI (GitHub Pages)
├── railway.toml        # One-click Railway deploy
└── requirements.txt
```

---

## Step 1 — Train on Google Colab (free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `train_colab.py`
3. Runtime → Change runtime type → **T4 GPU**
4. Fill in your config at the top of the file:
```python
"hf_token" : "hf_xxxx",              # Hugging Face write token
"hf_repo"  : "your-username/IP-AIv1" # Your HF username
```
5. Run All — takes ~3-4 hours
6. Model auto-uploads to Hugging Face when done

---

## Step 2 — Deploy API (free on Railway)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your `IP-AIv1` repo
4. Railway reads `railway.toml` automatically
5. API live at `https://your-app.railway.app`

---

## Step 3 — Chat UI (free on GitHub Pages)

1. In your repo → Settings → Pages → Source: main / (root)
2. UI live at `https://yourusername.github.io/IP-AIv1`
3. Enter your Railway URL in the UI → Connect

---

## API Reference

### Chat (Groq-compatible)
```bash
curl -X POST https://your-api.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ip-ai-v1",
    "messages": [{"role": "user", "content": "What is the latest news?"}],
    "max_tokens": 512,
    "stream": false,
    "use_tools": true
  }'
```

### Streaming
```bash
curl -X POST https://your-api.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ip-ai-v1","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Direct web search (no model needed)
```bash
curl -X POST https://your-api.railway.app/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI news"}'
```

### Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health + model info |
| `/v1/models` | GET | List models |
| `/v1/chat/completions` | POST | Chat (Groq-compatible, streaming) |
| `/v1/generate` | POST | Raw generation |
| `/v1/tools` | GET | List available tools |
| `/v1/tools/search` | POST | Direct web search |
| `/v1/tools/wikipedia` | POST | Direct Wikipedia lookup |

---

## Architecture Details

| Component | Detail |
|---|---|
| Type | Causal Transformer (GPT-style) |
| Parameters | ~50M |
| Layers | 12 |
| Dimensions | 768 |
| Query heads | 12 |
| KV heads | 4 (GQA) |
| Activation | SwiGLU |
| Norm | RMSNorm |
| Position | RoPE |
| Context | 1024 tokens |
| Tokenizer | BPE (32k vocab) |
| Pretrain data | OpenWebText (100k docs) |
| Finetune data | Alpaca (instruction following) |

---

## Internet Access

IP-AIv1 has real-time internet via three built-in tools:

- **web_search** — DuckDuckGo (no API key needed)
- **wikipedia_search** — Live Wikipedia summaries
- **fetch_page** — Read any URL

The model learns to call these tools during inference when it needs current information.

---

Built by Yash — IP-AIv1 is a fully independent open model project.
