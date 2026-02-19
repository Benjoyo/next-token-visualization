<div align="center">

# LLM Next-Token Visualization

![Python](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Transformers](https://img.shields.io/badge/LLM-Hugging%20Face%20Transformers-FFB000)
![Run with uv](https://img.shields.io/badge/run-uv-5C56F7)

**What happens to next-token probabilities when you change one knob?**

**🧠 Visualize token-by-token sampling with chat templates, nucleus filtering, constrained decoding, and attribution heatmaps in one local app ⚡**

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [API](#-api)

</div>

## 🎯 Overview

Sampling is usually a **black box**. You tweak temperature or top-p and hope for better text, without seeing what changed at the token level.

**The Solution:** This demo exposes every generation step: prompt/completion tokens, pre/post-sampling probabilities, nucleus pruning, forced tokens, and token attribution to input context.

**The Result:** You can explain each generated token with concrete evidence, in a browser, with a backend you can read in one file.

## ✨ Features

- Visible special tokens across prompt, completion stack, and distribution bars.
- Chat-template-aware tokenization (when the model tokenizer provides a template).
- Side-by-side probability display:
  - `p_base`: raw model probability.
  - `p_final`: probability after temperature, penalties, constraints, and nucleus filtering.
- Pre-nucleus top-k rendering where excluded tokens remain visible (`kept = false`, `p_final = 0`).
- Click-to-force token from distribution, or force arbitrary text append.
- Editable completion path:
  - Click last completion token to delete.
  - Pick alternate candidate on historical token overlays to rewind and branch.
- Branching workflow (fork prompt/completion state and compare outcomes).
- Attribution heatmaps on hover for generated tokens and candidate next tokens.
- Attention attribution CPU fallback when attention scores are unavailable on active device.
- LLM on/off toggle that greys out model effects and clears distribution output.
- Constrained decoding using Outlines + llguidance:
  - Multiple choice
  - Regex
  - CFG
  - JSON schema

## 🚀 Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed.

### Run locally

```bash
uv run server.py
```

Then open:

```text
http://127.0.0.1:8000
```

### Try it

1. Wait for the default model to finish loading (`Qwen/Qwen2.5-0.5B-Instruct` unless overridden).
2. Type a prompt in `Input 1`.
3. Click `Step` to append one token at a time.
4. Hover generated tokens or distribution candidates to inspect attribution.
5. Change temperature/top-p and watch `p_final` shift in real time.

## 🧠 How It Works

### Sampling pipeline (per step)

1. Build model input IDs from prompt + completion IDs.
2. Optionally wrap messages with tokenizer chat template (`system` + `user`) and `add_generation_prompt=True`.
3. Compute next-token logits.
4. Apply repetition, presence, and frequency penalties.
5. Apply temperature.
6. Optionally apply constrained decoding mask.
7. Compute distribution:
   - Display top-k from pre-nucleus probabilities.
   - Compute nucleus keep-set from post-constraint distribution.
   - Set `p_final = 0` for tokens outside nucleus (or invalid under constraints).
8. Select token via greedy, sampling, forced token ID, or forced text append.

### Attribution pipeline

- Methods: `attention`, `saliency`, `input_x_gradient`, `integrated_gradients`.
- Attribution targets one completion token at a time.
- If attention attribution is unsupported/unavailable on GPU/MPS, the server retries on CPU automatically.

## 🎛️ Generation Controls

Default decoding params:

```json
{
  "greedy": false,
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "seed": null,
  "stop_sequences": []
}
```

Presets in UI: `greedy`, `balanced`, `reliable`, `creative`, `custom`.

## 🔌 API

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/model/status` | Read current model/device/template status |
| `POST` | `/api/model/load` | Load a Hugging Face model by repo ID |
| `POST` | `/api/preview` | Compute prompt/completion tokens + next-token distribution |
| `POST` | `/api/step` | Append one token (sampled/greedy/forced) or force text |
| `POST` | `/api/attribution` | Compute token-level attribution scores |

### Request shapes

`POST /api/model/load`

```json
{ "model_id": "Qwen/Qwen2.5-0.5B-Instruct" }
```

`POST /api/preview` / `POST /api/step`

```json
{
  "branch_id": "default",
  "prompt": "The capital of France is",
  "system_prompt": "You are a helpful assistant.",
  "use_chat_template": true,
  "completion_ids": [],
  "top_k": 10,
  "params": {
    "greedy": false,
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": null,
    "stop_sequences": []
  },
  "constraint": {
    "enabled": false,
    "type": "multiple_choice",
    "schema": "Yes\nNo\nMaybe"
  }
}
```

Additional `POST /api/step` fields:

```json
{
  "reset_rng": false,
  "force_token_id": null,
  "force_text": null
}
```

`POST /api/attribution`

```json
{
  "branch_id": "default",
  "prompt": "The capital of France is",
  "system_prompt": "You are a helpful assistant.",
  "use_chat_template": true,
  "completion_ids": [151645, 271],
  "target_index": 1,
  "target_token_id": 271,
  "method": "attention"
}
```

### Key response fields

- `prompt_tokens`, `completion_tokens`: token pills used in UI (with `special: true|false`).
- `dist.tokens[].p_base`: raw probability.
- `dist.tokens[].p_final`: post-sampling probability (0 when excluded).
- `dist.tokens[].kept`: nucleus inclusion flag.
- `dist.tokens[].valid`: constraint validity flag.
- `appended_meta`: per-appended-token snapshot (distribution + selection flags at append time).

## 🧱 Project Structure

```text
.
├── index.html   # Entire frontend (HTML/CSS/JS), no build step
└── server.py    # FastAPI server + model loading + sampling/attribution APIs
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_DEMO_DEFAULT_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model loaded on startup |
| `LLM_DEMO_LOG_LEVEL` | `info` | Server log level |

## 💾 LocalStorage Keys

Persisted UI state:

- `llmDemo.params`
- `llmDemo.modelId`
- `llmDemo.llmEnabled`
- `llmDemo.decodingCollapsed`
- `llmDemo.stepSpeedCollapsed`
- `llmDemo.attrCollapsed`
- `llmDemo.attrMethod`
- `llmDemo.constraintEnabled`
- `llmDemo.constraintType`
- `llmDemo.constraintSchema`
- `llmDemo.constraintCollapsed`
- `llmDemo.topK`
- `llmDemo.stepSpeedMs`