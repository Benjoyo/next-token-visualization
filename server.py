#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.110",
#   "uvicorn>=0.30",
#   "transformers>=4.41",
#   "torch",
#   "safetensors",
#   "accelerate",
#   "numpy",
#   "sentencepiece",
#   "protobuf",
# ]
# ///
"""
LLM Sampling Visualization Demo Server

Run:
  uv run server.py

Then open:
  http://127.0.0.1:8000

Notes:
- Default model: Qwen/Qwen2.5-0.5B-Instruct
- Supports loading any Hugging Face repository ID at runtime.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.environ.get("LLM_DEMO_DEFAULT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _preferred_dtype(device: str) -> torch.dtype:
    # Keep it simple and safe.
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def _visualize_token_piece(s: str) -> str:
    """
    Produce a human-friendly rendering for token pieces with leading whitespace/newlines.

    Examples:
      " Paris" -> "␠Paris"
      "\n" -> "⏎"
      "  " -> "␠␠"
    """
    if s == "":
        return "∅"
    if s == "\n":
        return "⏎"
    if s == "\t":
        return "⇥"
    # Replace leading spaces with visible markers
    i = 0
    while i < len(s) and s[i] == " ":
        i += 1
    if i > 0:
        return ("␠" * i) + s[i:]
    # Replace pure whitespace (rare)
    if s.strip() == "":
        return s.replace(" ", "␠").replace("\n", "⏎").replace("\t", "⇥")
    return s.replace("\n", "⏎")


def _token_pills_from_ids(tokenizer, ids: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for tid in ids:
        try:
            if tid in special_ids:
                raw = tokenizer.convert_ids_to_tokens(int(tid))
            else:
                raw = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        except Exception:
            raw = f"<id:{tid}>"
        out.append(
            {
                "id": int(tid),
                "text": raw,
                "display": _visualize_token_piece(raw),
            }
        )
    return out


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: List[int], penalty: float) -> torch.Tensor:
    if penalty is None or float(penalty) == 1.0:
        return logits
    penalty = float(penalty)
    # This mirrors transformers' RepetitionPenaltyLogitsProcessor.
    unique_ids = set(int(i) for i in input_ids)
    for tid in unique_ids:
        if tid < 0 or tid >= logits.numel():
            continue
        val = logits[tid]
        logits[tid] = torch.where(val < 0, val * penalty, val / penalty)
    return logits


def _apply_presence_frequency_penalties(
    logits: torch.Tensor,
    input_ids: List[int],
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    pp = float(presence_penalty or 0.0)
    fp = float(frequency_penalty or 0.0)
    if pp == 0.0 and fp == 0.0:
        return logits
    counts: Dict[int, int] = {}
    for tid in input_ids:
        tid = int(tid)
        counts[tid] = counts.get(tid, 0) + 1
    for tid, c in counts.items():
        if tid < 0 or tid >= logits.numel():
            continue
        if pp != 0.0:
            logits[tid] = logits[tid] - pp
        if fp != 0.0:
            logits[tid] = logits[tid] - (fp * float(c))
    return logits


def _nucleus_filter_sorted(
    sorted_probs: torch.Tensor, sorted_ids: torch.Tensor, top_p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inputs must already be sorted in descending prob order.
    Returns filtered (ids, probs) renormalized.
    """
    top_p = float(top_p)
    if top_p >= 1.0:
        probs = sorted_probs
        ids = sorted_ids
        # Renormalize defensively
        s = probs.sum()
        if s > 0:
            probs = probs / s
        return ids, probs

    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= top_p
    # Always keep at least one token
    if mask.numel() > 0:
        mask[0] = True
    ids = sorted_ids[mask]
    probs = sorted_probs[mask]
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return ids, probs


@dataclass
class ModelState:
    model_id: str = ""
    device: str = "cpu"
    dtype: str = "float32"
    loading: bool = False
    ready: bool = False
    error: Optional[str] = None
    has_chat_template: bool = False
    eos_token_id: Optional[int] = None
    vocab_size: Optional[int] = None

    # Live objects (not in asdict)
    model: Any = None
    tokenizer: Any = None


STATE_LOCK = threading.Lock()
STATE = ModelState()


# RNG state per branch to make seeding meaningful across steps.
RNG_LOCK = threading.Lock()
RNGS: Dict[str, Dict[str, Any]] = {}


def _reset_rng(branch_id: str, seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    RNGS[branch_id] = {"seed": int(seed), "gen": gen}
    return gen


def _get_rng(branch_id: str, seed: Optional[int], reset: bool = False) -> Optional[torch.Generator]:
    if seed is None:
        return None
    with RNG_LOCK:
        entry = RNGS.get(branch_id)
        if reset or entry is None or int(entry.get("seed")) != int(seed):
            return _reset_rng(branch_id, int(seed))
        return entry["gen"]


def _clear_model_objects():
    global STATE
    if STATE.model is not None:
        try:
            del STATE.model
        except Exception:
            pass
    if STATE.tokenizer is not None:
        try:
            del STATE.tokenizer
        except Exception:
            pass
    STATE.model = None
    STATE.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_worker(model_id: str):
    global STATE
    with STATE_LOCK:
        STATE.loading = True
        STATE.ready = False
        STATE.error = None

    try:
        device = _select_device()
        dtype = _preferred_dtype(device)

        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

        # Some tokenizers don't have pad token; it's fine for single-step decoding.
        # But set it defensively to eos to avoid warnings.
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.eval()

        # Move to device
        if device == "cuda":
            model.to("cuda")
        elif device == "mps":
            model.to("mps")
        else:
            model.to("cpu")

        # "Loading status by testing tokenization"
        _ = tok("Hello!", return_tensors=None)

        has_chat_template = bool(getattr(tok, "chat_template", None))
        eos_token_id = tok.eos_token_id
        vocab_size = getattr(tok, "vocab_size", None) or getattr(model.config, "vocab_size", None)

        with STATE_LOCK:
            _clear_model_objects()
            STATE.model_id = model_id
            STATE.device = device
            STATE.dtype = str(dtype).replace("torch.", "")
            STATE.model = model
            STATE.tokenizer = tok
            STATE.loading = False
            STATE.ready = True
            STATE.error = None
            STATE.has_chat_template = has_chat_template
            STATE.eos_token_id = int(eos_token_id) if eos_token_id is not None else None
            STATE.vocab_size = int(vocab_size) if vocab_size is not None else None

    except Exception as e:
        tb = traceback.format_exc(limit=10)
        with STATE_LOCK:
            STATE.loading = False
            STATE.ready = False
            STATE.error = f"{e}\n\n{tb}"


def start_model_load(model_id: str):
    # Fire and forget background thread.
    t = threading.Thread(target=_load_model_worker, args=(model_id,), daemon=True)
    t.start()


def _require_ready_model() -> Tuple[Any, Any, ModelState]:
    with STATE_LOCK:
        if STATE.loading and not STATE.ready:
            raise HTTPException(status_code=503, detail="Model is loading. Try again shortly.")
        if not STATE.ready or STATE.model is None or STATE.tokenizer is None:
            msg = STATE.error or "Model is not ready."
            raise HTTPException(status_code=503, detail=msg)
        # Copy public state
        public = ModelState(
            model_id=STATE.model_id,
            device=STATE.device,
            dtype=STATE.dtype,
            loading=STATE.loading,
            ready=STATE.ready,
            error=STATE.error,
            has_chat_template=STATE.has_chat_template,
            eos_token_id=STATE.eos_token_id,
            vocab_size=STATE.vocab_size,
        )
        return STATE.model, STATE.tokenizer, public


def _build_input_ids(tokenizer, prompt: str, completion_ids: List[int]) -> Tuple[List[int], List[int]]:
    """
    Returns (prompt_ids_used_by_model, full_input_ids_used_by_model).
    - If the tokenizer has a chat template, the prompt is wrapped as a user message and
      add_generation_prompt=True is used, so completion_ids correspond to assistant tokens.
    - Otherwise prompt_ids are simply tokenizer(prompt) ids.
    """
    completion_ids = [int(x) for x in (completion_ids or [])]

    if bool(getattr(tokenizer, "chat_template", None)):
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        prompt_ids = [int(x) for x in prompt_ids]
    else:
        enc = tokenizer(prompt, add_special_tokens=True, return_tensors=None)
        prompt_ids = enc["input_ids"]
        # Some tokenizers return List[int], others return List[List[int]].
        if isinstance(prompt_ids, (list, tuple)) and len(prompt_ids) > 0 and isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = prompt_ids[0]
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.detach().cpu().tolist()
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = prompt_ids.tolist()
        prompt_ids = [int(x) for x in list(prompt_ids)]

    full_ids = prompt_ids + completion_ids
    return prompt_ids, full_ids


def _compute_next_token_distribution(
    model,
    tokenizer,
    full_input_ids: List[int],
    top_k: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> Dict[str, Any]:
    device = _select_device()
    # Trust current model device rather than re-selecting
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)

    top_k = int(max(1, min(int(top_k), 50)))
    temperature = float(temperature)
    top_p = float(top_p)

    input_tensor = torch.tensor([full_input_ids], device=model_device, dtype=torch.long)

    with torch.inference_mode():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits[0, -1]

    logits_base = logits.float()

    # Baseline probabilities (raw model)
    probs_base = torch.softmax(logits_base, dim=-1)

    # Apply decoding params to obtain sampling distribution
    logits_mod = logits_base.clone()
    logits_mod = _apply_repetition_penalty(logits_mod, full_input_ids, repetition_penalty)
    logits_mod = _apply_presence_frequency_penalties(logits_mod, full_input_ids, presence_penalty, frequency_penalty)

    # Temperature
    temp = float(temperature)
    if temp <= 0:
        temp = 1e-6
    logits_temp = logits_mod / temp
    probs_temp = torch.softmax(logits_temp, dim=-1)

    # Nucleus filtering (top-p)
    sorted_probs, sorted_ids = torch.sort(probs_temp, descending=True)
    kept_ids, kept_probs = _nucleus_filter_sorted(sorted_probs, sorted_ids, top_p)

    # Take the first K for display (these are the highest-prob tokens in the final distribution)
    k = min(top_k, int(kept_ids.numel()))
    top_ids = kept_ids[:k]
    top_final_probs = kept_probs[:k]
    top_base_probs = probs_base[top_ids]

    # "Other mass" = everything beyond top-k in the *final* distribution.
    other_final = float(max(0.0, 1.0 - float(top_final_probs.sum().item())))
    other_base = float(max(0.0, 1.0 - float(top_base_probs.sum().item())))

    # Decode tokens for display
    top_ids_list = [int(x) for x in top_ids.detach().cpu().tolist()]
    top_tokens = _token_pills_from_ids(tokenizer, top_ids_list)

    # Attach probabilities
    for i, tok in enumerate(top_tokens):
        tok["p_base"] = float(top_base_probs[i].detach().cpu().item())
        tok["p_final"] = float(top_final_probs[i].detach().cpu().item())

    return {
        "top_k": top_k,
        "tokens": top_tokens,
        "other_p_base": other_base,
        "other_p_final": other_final,
        "kept_token_count": int(kept_ids.numel()),
    }


def _choose_next_token(
    tokenizer,
    dist: Dict[str, Any],
    greedy: bool,
    branch_id: str,
    seed: Optional[int],
    reset_rng: bool,
) -> int:
    # dist["tokens"] are top-k in the final distribution, but sampling must happen across nucleus set.
    # We'll sample using the kept distribution reconstructed by re-running nucleus filter quickly.
    # For simplicity, we use the top-k set for greedy and for sampling when top_k is small.
    # But to be correct, we should sample from the full nucleus set.

    # We'll store nucleus info in dist? Currently we only store count.
    # We'll re-compute needed tensors: we can’t without logits/probs.
    # Instead, accept slight approximation by sampling from displayed top-k.
    #
    # For educational demos this is acceptable, but we'll do better by returning kept_ids/probs is too big.
    #
    # Better: in step endpoint we compute sampling from kept_ids/kept_probs directly, not via dist.
    raise RuntimeError("Internal error: _choose_next_token should not be called.")


class DecodingParams(BaseModel):
    greedy: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)


class PreviewRequest(BaseModel):
    branch_id: str = "default"
    prompt: str = ""
    completion_ids: List[int] = Field(default_factory=list)
    top_k: int = 10
    params: DecodingParams = Field(default_factory=DecodingParams)


class StepRequest(BaseModel):
    branch_id: str = "default"
    reset_rng: bool = False
    prompt: str = ""
    completion_ids: List[int] = Field(default_factory=list)
    top_k: int = 10
    params: DecodingParams = Field(default_factory=DecodingParams)
    force_token_id: Optional[int] = None
    force_text: Optional[str] = None


app = FastAPI(title="LLM Sampling Visualization Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # convenient for local demos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    if not INDEX_HTML.exists():
        # We'll still serve a minimal page if index.html isn't found.
        pass
    start_model_load(DEFAULT_MODEL_ID)


@app.get("/", response_class=HTMLResponse)
def index():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return HTMLResponse(
        "<h1>index.html not found</h1><p>Place index.html next to server.py.</p>",
        status_code=200,
    )


@app.get("/api/model/status")
def model_status():
    with STATE_LOCK:
        d = asdict(
            ModelState(
                model_id=STATE.model_id,
                device=STATE.device,
                dtype=STATE.dtype,
                loading=STATE.loading,
                ready=STATE.ready,
                error=STATE.error,
                has_chat_template=STATE.has_chat_template,
                eos_token_id=STATE.eos_token_id,
                vocab_size=STATE.vocab_size,
            )
        )
    return d


class LoadModelRequest(BaseModel):
    model_id: str


@app.post("/api/model/load")
def load_model(req: LoadModelRequest):
    model_id = (req.model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    with STATE_LOCK:
        if STATE.loading:
            raise HTTPException(status_code=409, detail="A model is already loading.")
        # Mark as loading immediately
        STATE.loading = True
        STATE.ready = False
        STATE.error = None
    start_model_load(model_id)
    return {"ok": True, "model_id": model_id}


@app.post("/api/preview")
def preview(req: PreviewRequest):
    model, tok, public = _require_ready_model()

    prompt_ids, full_ids = _build_input_ids(tok, req.prompt, req.completion_ids)

    dist = _compute_next_token_distribution(
        model=model,
        tokenizer=tok,
        full_input_ids=full_ids,
        top_k=req.top_k,
        temperature=req.params.temperature,
        top_p=req.params.top_p,
        repetition_penalty=req.params.repetition_penalty,
        presence_penalty=req.params.presence_penalty,
        frequency_penalty=req.params.frequency_penalty,
    )

    # Display tokens (prompt and completion separately, without chat framing)
    prompt_display_ids = tok(req.prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
    if isinstance(prompt_display_ids, (list, tuple)) and len(prompt_display_ids) > 0 and isinstance(prompt_display_ids[0], (list, tuple)):
        prompt_display_ids = prompt_display_ids[0]
    if isinstance(prompt_display_ids, torch.Tensor):
        prompt_display_ids = prompt_display_ids.detach().cpu().tolist()
    if isinstance(prompt_display_ids, np.ndarray):
        prompt_display_ids = prompt_display_ids.tolist()
    prompt_display_ids = [int(x) for x in list(prompt_display_ids)]
    prompt_tokens = _token_pills_from_ids(tok, prompt_display_ids)
    completion_tokens = _token_pills_from_ids(tok, [int(x) for x in (req.completion_ids or [])])

    return {
        "model": {
            "model_id": public.model_id,
            "device": public.device,
            "dtype": public.dtype,
            "has_chat_template": public.has_chat_template,
            "eos_token_id": public.eos_token_id,
            "vocab_size": public.vocab_size,
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "context_token_count": int(len(full_ids)),
        "dist": dist,
    }


@app.post("/api/step")
def step(req: StepRequest):
    model, tok, public = _require_ready_model()

    # Forcing multiple tokens via text is treated as a pure append (no sampling).
    if req.force_text is not None and req.force_text != "":
        forced_ids = tok(req.force_text, add_special_tokens=False, return_tensors=None)["input_ids"]
        if isinstance(forced_ids, (list, tuple)) and len(forced_ids) > 0 and isinstance(forced_ids[0], (list, tuple)):
            forced_ids = forced_ids[0]
        if isinstance(forced_ids, torch.Tensor):
            forced_ids = forced_ids.detach().cpu().tolist()
        if isinstance(forced_ids, np.ndarray):
            forced_ids = forced_ids.tolist()
        forced_ids = [int(x) for x in list(forced_ids)]
        new_completion = [int(x) for x in (req.completion_ids or [])] + forced_ids

        # Still return the distribution at the point of forcing (current state)
        prompt_ids, full_ids = _build_input_ids(tok, req.prompt, req.completion_ids)
        dist = _compute_next_token_distribution(
            model=model,
            tokenizer=tok,
            full_input_ids=full_ids,
            top_k=req.top_k,
            temperature=req.params.temperature,
            top_p=req.params.top_p,
            repetition_penalty=req.params.repetition_penalty,
            presence_penalty=req.params.presence_penalty,
            frequency_penalty=req.params.frequency_penalty,
        )

        appended_tokens = _token_pills_from_ids(tok, forced_ids)

        # Stop detection
        eos = public.eos_token_id
        stop_reason = None
        done = False
        if eos is not None and eos in forced_ids:
            stop_reason = "eos_token"
            done = True
        # stop sequences in decoded completion
        decoded_completion = tok.decode(new_completion, clean_up_tokenization_spaces=False)
        for s in (req.params.stop_sequences or []):
            if s and s in decoded_completion:
                stop_reason = "stop_sequence"
                done = True
                break

        return {
            "model": {
                "model_id": public.model_id,
                "device": public.device,
                "dtype": public.dtype,
                "has_chat_template": public.has_chat_template,
                "eos_token_id": public.eos_token_id,
                "vocab_size": public.vocab_size,
            },
            "dist": dist,
            "selected": None,
            "appended": appended_tokens,
            "new_completion_ids": new_completion,
            "done": done,
            "stop_reason": stop_reason,
        }

    # Normal step: compute distribution, then choose token (greedy / sampled / forced token id)
    prompt_ids, full_ids = _build_input_ids(tok, req.prompt, req.completion_ids)

    # Forward pass to get logits and sampling distribution. We'll reuse the same computation for dist and sampling.
    device = _select_device()
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)

    input_tensor = torch.tensor([full_ids], device=model_device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits[0, -1].float()

    logits_base = logits
    probs_base = torch.softmax(logits_base, dim=-1)

    logits_mod = logits_base.clone()
    logits_mod = _apply_repetition_penalty(logits_mod, full_ids, req.params.repetition_penalty)
    logits_mod = _apply_presence_frequency_penalties(
        logits_mod, full_ids, req.params.presence_penalty, req.params.frequency_penalty
    )

    temp = float(req.params.temperature)
    if temp <= 0:
        temp = 1e-6
    probs_temp = torch.softmax(logits_mod / temp, dim=-1)

    sorted_probs, sorted_ids = torch.sort(probs_temp, descending=True)
    kept_ids, kept_probs = _nucleus_filter_sorted(sorted_probs, sorted_ids, float(req.params.top_p))

    # Distribution for display (top-k of final distribution)
    top_k = int(max(1, min(int(req.top_k), 50)))
    k = min(top_k, int(kept_ids.numel()))
    top_ids = kept_ids[:k]
    top_final_probs = kept_probs[:k]
    top_base_probs = probs_base[top_ids]
    other_final = float(max(0.0, 1.0 - float(top_final_probs.sum().item())))
    other_base = float(max(0.0, 1.0 - float(top_base_probs.sum().item())))

    top_ids_list = [int(x) for x in top_ids.detach().cpu().tolist()]
    top_tokens = _token_pills_from_ids(tok, top_ids_list)
    for i, t in enumerate(top_tokens):
        t["p_base"] = float(top_base_probs[i].detach().cpu().item())
        t["p_final"] = float(top_final_probs[i].detach().cpu().item())

    dist = {
        "top_k": top_k,
        "tokens": top_tokens,
        "other_p_base": other_base,
        "other_p_final": other_final,
        "kept_token_count": int(kept_ids.numel()),
    }

    # Choose token
    selected_id: int
    selected_prob_final: Optional[float] = None

    if req.force_token_id is not None:
        selected_id = int(req.force_token_id)
        selected_prob_final = None
    else:
        if bool(req.params.greedy):
            selected_id = int(kept_ids[0].detach().cpu().item())
            selected_prob_final = float(kept_probs[0].detach().cpu().item())
        else:
            gen = _get_rng(req.branch_id, req.params.seed, reset=bool(req.reset_rng))
            # Sample from nucleus set
            kept_probs_cpu = kept_probs.detach().cpu()
            kept_ids_cpu = kept_ids.detach().cpu()
            # torch.multinomial expects 1D probs
            idx = int(torch.multinomial(kept_probs_cpu, num_samples=1, replacement=False, generator=gen).item())
            selected_id = int(kept_ids_cpu[idx].item())
            selected_prob_final = float(kept_probs_cpu[idx].item())

    appended_ids = [selected_id]
    new_completion = [int(x) for x in (req.completion_ids or [])] + appended_ids
    appended_tokens = _token_pills_from_ids(tok, appended_ids)

    # Stop detection
    eos = public.eos_token_id
    stop_reason = None
    done = False
    if eos is not None and selected_id == int(eos):
        stop_reason = "eos_token"
        done = True
    else:
        decoded_completion = tok.decode(new_completion, clean_up_tokenization_spaces=False)
        for s in (req.params.stop_sequences or []):
            if s and s in decoded_completion:
                stop_reason = "stop_sequence"
                done = True
                break

    selected_info = {
        "id": int(selected_id),
        "text": appended_tokens[0]["text"],
        "display": appended_tokens[0]["display"],
        "p_final": selected_prob_final,
    }

    return {
        "model": {
            "model_id": public.model_id,
            "device": public.device,
            "dtype": public.dtype,
            "has_chat_template": public.has_chat_template,
            "eos_token_id": public.eos_token_id,
            "vocab_size": public.vocab_size,
        },
        "dist": dist,
        "selected": selected_info,
        "appended": appended_tokens,
        "new_completion_ids": new_completion,
        "done": done,
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
