# LLM Sampling Demo – Agent Notes

## Quick start
- Run the demo server: `uv run server.py`
- Open: `http://127.0.0.1:8000`
- `server.py` is a single-file uv script with embedded dependencies (no separate venv setup).

## Project layout
- `index.html`: **all** UI (HTML/CSS/JS). No build step.
- `server.py`: FastAPI backend that loads the HF model and serves `/` + JSON APIs.

## Key behaviors to preserve
- **Special tokens must be visible**: the server marks tokens with `special: true`, and the UI colors them distinctly.
- **Chat templates**: the prompt is tokenized with the model’s chat template when available; template tokens are shown in the token lists.
- **Distribution rendering**:
  - The server returns top‑k *pre‑nucleus* tokens.
  - Tokens excluded by nucleus sampling are still returned with `p_final = 0` and `kept = false` so the UI can grey them out.
- **Attribution heatmap**: hovering a generated token (stack) or a candidate token (distribution) highlights input tokens with blue alpha; attribution methods are selected from the sidebar.
- **Attention attribution fallback**: if attention outputs aren’t available on the active device, the server retries attention attribution on CPU.
- **LLM on/off**: toggling the LLM greys out the box and clears the distribution column.
- **Completion editing**: clicking the last completion token deletes it; choosing an alternative token rewinds later tokens.

## Server API
- `GET /api/model/status`
- `POST /api/model/load` `{ model_id }`
- `POST /api/preview` `{ prompt, completion_ids, top_k, params }`
- `POST /api/step` `{ prompt, completion_ids, top_k, params, force_token_id?, force_text? }`
- `POST /api/attribution` `{ prompt, completion_ids, target_index, target_token_id, method }`

The preview/step responses include:
- `prompt_tokens`, `completion_tokens`, `dist`
- `appended_meta` for forced tokens (distribution + flags at the time of append)

## UI conventions
- Use CSS variables in `:root` for colors; buttons should **not** use gradients.
- No box shadows (the base CSS wipes them, but avoid adding new ones).
- SVG “wires” are drawn in `drawWires()` and should scale with layout changes.

## LocalStorage
Persisted keys (keep in sync with UI state):
- `llmDemo.params` – last used decoding params
- `llmDemo.modelId` – last HF model ID
- `llmDemo.llmEnabled` – LLM on/off
- `llmDemo.decodingCollapsed` – decoding panel collapsed
- `llmDemo.stepSpeedCollapsed` – speed panel collapsed
- `llmDemo.attrCollapsed` – attribution panel collapsed
- `llmDemo.attrMethod` – attribution method
- `llmDemo.topK` – top‑k for distribution
- `llmDemo.stepSpeedMs` – autoplay speed

Decoding params are **per-branch** in UI (`b.params`), but the stored defaults apply to new branches.

## Tips
- Keep `index.html` readable; add small comments only for non‑obvious logic.
- If you change the server response shape, update the UI parsing accordingly.
