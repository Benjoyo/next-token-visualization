#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.110",
#   "uvicorn>=0.30",
#   "transformers>=4.41,<5.0.0",
#   "torch",
#   "safetensors",
#   "accelerate",
#   "numpy",
#   "sentencepiece",
#   "protobuf",
#   "inseq",
#   "outlines>=1.0.0",
#   "llguidance",
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
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
from inseq import load_model as inseq_load_model
from outlines import from_transformers as outlines_from_transformers
from outlines.generator import Generator as outlines_generator
import outlines.types as outlines_types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.environ.get("LLM_DEMO_DEFAULT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LOG_LEVEL = os.environ.get("LLM_DEMO_LOG_LEVEL", "info").lower()
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("llm_demo")


def _patch_captum_ig_mps():
    try:
        import captum.attr._utils.approximation_methods as am
    except Exception:
        return
    if getattr(am, "_llm_demo_mps_patched", False):
        return
    if not hasattr(am, "gauss_legendre_builders"):
        return
    orig = am.gauss_legendre_builders

    def _patched_gauss_legendre_builders():
        step_sizes, alphas = orig()

        def step_sizes_fp32(n: int):
            return [float(x) for x in step_sizes(n)]

        def alphas_fp32(n: int):
            return [float(x) for x in alphas(n)]

        return step_sizes_fp32, alphas_fp32

    am.gauss_legendre_builders = _patched_gauss_legendre_builders
    am._llm_demo_mps_patched = True


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
    special_texts = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for tid in ids:
        try:
            if tid in special_ids:
                raw = tokenizer.convert_ids_to_tokens(int(tid))
            else:
                raw = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        except Exception:
            raw = f"<id:{tid}>"
        is_special = (tid in special_ids) or (raw in special_texts)
        out.append(
            {
                "id": int(tid),
                "text": raw,
                "display": _visualize_token_piece(raw),
                "special": bool(is_special),
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

ATTR_LOCK = threading.Lock()
ATTR_MODEL = None
ATTR_MODEL_ID: Optional[str] = None
ATTR_MODEL_CPU = None
ATTR_MODEL_CPU_ID: Optional[str] = None
ATTR_TOKENIZER_CPU = None

CONSTRAINT_TYPES = {"multiple_choice", "regex", "cfg", "json_schema"}


@dataclass
class ConstraintSpec:
    kind: str
    schema: str
    output_type: Any


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
    global STATE, ATTR_MODEL, ATTR_MODEL_ID, ATTR_MODEL_CPU, ATTR_MODEL_CPU_ID, ATTR_TOKENIZER_CPU
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
    with ATTR_LOCK:
        ATTR_MODEL = None
        ATTR_MODEL_ID = None
        ATTR_MODEL_CPU = None
        ATTR_MODEL_CPU_ID = None
        ATTR_TOKENIZER_CPU = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_pad_token(tokenizer, model=None) -> bool:
    if tokenizer.pad_token_id is not None:
        return True
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer.pad_token_id is not None
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer.pad_token_id is not None
    try:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if model is not None:
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
    except Exception:
        return False
    return tokenizer.pad_token_id is not None


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

        # Some tokenizers don't have pad token; set it defensively.
        _ensure_pad_token(tok)

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


def _build_input_ids(
    tokenizer,
    prompt: str,
    completion_ids: List[int],
    use_chat_template: Optional[bool] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """
    Returns (prompt_ids_used_by_model, full_input_ids_used_by_model).
    - If a chat template is available and enabled, the prompt is wrapped as a user message
      (and system message, if provided) with add_generation_prompt=True so completion_ids
      correspond to assistant tokens.
    - Otherwise prompt_ids are simply tokenizer(prompt) ids.
    """
    completion_ids = [int(x) for x in (completion_ids or [])]

    has_template = bool(getattr(tokenizer, "chat_template", None))
    use_template = has_template and (use_chat_template is not False)
    if use_template:
        messages: List[Dict[str, str]] = []
        sys_msg = (system_prompt or "").strip()
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})
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


def _decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def _ids_to_tokens(tokenizer, ids: List[int]) -> List[str]:
    out: List[str] = []
    for tid in ids:
        try:
            out.append(tokenizer.convert_ids_to_tokens(int(tid)))
        except Exception:
            out.append(str(tid))
    return out


def _tokens_to_string(tokenizer, tokens: List[str]) -> str:
    try:
        return tokenizer.convert_tokens_to_string(tokens)
    except Exception:
        try:
            ids = tokenizer.convert_tokens_to_ids(tokens)
            return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            return "".join(tokens)


def _sanitize_floats(vals: List[float]) -> List[float]:
    out: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if not math.isfinite(fv):
            fv = 0.0
        out.append(fv)
    return out


def _get_attr_model(method: str):
    global ATTR_MODEL, ATTR_MODEL_ID
    model, tok, public = _require_ready_model()
    _ensure_pad_token(tok, model=model)
    with ATTR_LOCK:
        if ATTR_MODEL is None or ATTR_MODEL_ID != public.model_id:
            try:
                model.config.output_attentions = True
            except Exception:
                pass
            try:
                model.config.output_hidden_states = True
            except Exception:
                pass
            ATTR_MODEL = inseq_load_model(model, "saliency", tokenizer=tok, device=public.device)
            ATTR_MODEL_ID = public.model_id
    if ATTR_MODEL is not None:
        attr_tok = getattr(ATTR_MODEL, "tokenizer", None)
        attr_model_ref = getattr(ATTR_MODEL, "model", None)
        if attr_tok is not None:
            _ensure_pad_token(attr_tok, model=attr_model_ref)
    return ATTR_MODEL, tok, public


def _get_attr_model_cpu(model_id: str):
    global ATTR_MODEL_CPU, ATTR_MODEL_CPU_ID, ATTR_TOKENIZER_CPU
    with ATTR_LOCK:
        if ATTR_MODEL_CPU is None or ATTR_MODEL_CPU_ID != model_id:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=True,
            )
            _ensure_pad_token(tok)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model.eval()
            try:
                model.config.output_attentions = True
            except Exception:
                pass
            if hasattr(model, "set_attn_implementation"):
                try:
                    model.set_attn_implementation("eager")
                except Exception:
                    pass
            ATTR_MODEL_CPU = inseq_load_model(model, "saliency", tokenizer=tok, device="cpu")
            ATTR_MODEL_CPU_ID = model_id
            ATTR_TOKENIZER_CPU = tok
            attr_tok = getattr(ATTR_MODEL_CPU, "tokenizer", None)
            if attr_tok is not None:
                _ensure_pad_token(attr_tok, model=getattr(ATTR_MODEL_CPU, "model", None))
    return ATTR_MODEL_CPU, ATTR_TOKENIZER_CPU


def _parse_multiple_choice_items(schema_text: str) -> List[str]:
    text = str(schema_text or "")
    stripped = text.strip()
    parsed_items: List[str] = []

    # Accept either newline-separated options or a JSON list for convenience.
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                parsed_items = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            parsed_items = []

    if not parsed_items:
        parsed_items = [line.strip() for line in text.splitlines() if line.strip()]

    deduped: List[str] = []
    seen = set()
    for item in parsed_items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _resolve_constraint_spec(constraint: Optional["ConstraintConfig"]) -> Optional[ConstraintSpec]:
    if constraint is None or not bool(constraint.enabled):
        return None

    kind = str(constraint.type or "").strip().lower()
    if kind not in CONSTRAINT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported constrained sampling type.")

    schema_text = str(constraint.schema_text or "")

    try:
        if kind == "multiple_choice":
            items = _parse_multiple_choice_items(schema_text)
            if not items:
                raise ValueError("Provide at least one choice (one per line or JSON list).")
            output_type = outlines_types.Choice(items)
            normalized_schema = "\n".join(items)
        elif kind == "regex":
            pattern = schema_text.strip()
            if not pattern:
                raise ValueError("Regex pattern is required.")
            output_type = outlines_types.Regex(pattern)
            normalized_schema = pattern
        elif kind == "cfg":
            grammar = schema_text.strip()
            if not grammar:
                raise ValueError("CFG grammar is required.")
            output_type = outlines_types.CFG(grammar)
            normalized_schema = grammar
        else:
            json_text = schema_text.strip()
            if not json_text:
                raise ValueError("JSON schema is required.")
            json_schema_obj = json.loads(json_text)
            output_type = outlines_types.JsonSchema(json_schema_obj)
            normalized_schema = json.dumps(json_schema_obj, ensure_ascii=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid constrained sampling schema: {e}")

    return ConstraintSpec(kind=kind, schema=normalized_schema, output_type=output_type)


def _build_constraint_logits_processor(model, tokenizer, spec: ConstraintSpec):
    try:
        outlines_model = outlines_from_transformers(model, tokenizer)
        generator = outlines_generator(outlines_model, spec.output_type)
        processor = getattr(generator, "logits_processor", None)
        if processor is None:
            raise RuntimeError("Outlines did not return a logits processor.")
        if hasattr(processor, "reset"):
            processor.reset()
        return processor
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize constrained sampling ({spec.kind}): {e}",
        )


def _apply_constraint_logits_for_context(
    logits: torch.Tensor,
    processor: Any,
    prompt_ids: List[int],
    completion_ids: List[int],
) -> torch.Tensor:
    # Outlines processors are stateful; we replay prior completion tokens
    # so the mask reflects the next token at the current position.
    full_ids = [int(x) for x in (prompt_ids + completion_ids)]
    prompt_ids = [int(x) for x in (prompt_ids or [])]
    completion_ids = [int(x) for x in (completion_ids or [])]

    logits_batch = logits.unsqueeze(0)
    device = logits.device

    if len(completion_ids) == 0:
        first_input = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        return processor(first_input, logits_batch)[0]

    # First generation step: initialize processor state without consuming tokens.
    init_input = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    _ = processor(init_input, logits_batch.clone())

    # Replay previously generated completion tokens except the latest one.
    for i in range(1, len(completion_ids)):
        prefix_input = torch.tensor([prompt_ids + completion_ids[:i]], device=device, dtype=torch.long)
        _ = processor(prefix_input, logits_batch.clone())

    # Final call advances the latest completion token and returns next-token mask.
    full_input = torch.tensor([full_ids], device=device, dtype=torch.long)
    return processor(full_input, logits_batch)[0]


def _compute_sampling_state(
    model,
    tokenizer,
    prompt_ids: List[int],
    completion_ids: List[int],
    top_k: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    constraint: Optional[ConstraintSpec] = None,
) -> Dict[str, Any]:
    device = _select_device()
    # Trust current model device rather than re-selecting
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)

    prompt_ids = [int(x) for x in (prompt_ids or [])]
    completion_ids = [int(x) for x in (completion_ids or [])]
    full_input_ids = prompt_ids + completion_ids

    top_k = int(max(1, min(int(top_k), 50)))
    top_p = float(top_p)

    input_tensor = torch.tensor([full_input_ids], device=model_device, dtype=torch.long)

    with torch.inference_mode():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits[0, -1]

    logits_base = logits.float()
    probs_base = torch.softmax(logits_base, dim=-1)

    logits_mod = logits_base.clone()
    logits_mod = _apply_repetition_penalty(logits_mod, full_input_ids, repetition_penalty)
    logits_mod = _apply_presence_frequency_penalties(
        logits_mod, full_input_ids, presence_penalty, frequency_penalty
    )

    temp = float(temperature)
    if temp <= 0:
        temp = 1e-6
    logits_temp = logits_mod / temp
    probs_temp_unconstrained = torch.softmax(logits_temp, dim=-1)

    # Display top-k from full pre-nucleus, pre-constraint distribution.
    sorted_probs_display, sorted_ids_display = torch.sort(probs_temp_unconstrained, descending=True)

    logits_for_sampling = logits_temp
    valid_mask: Optional[torch.Tensor] = None
    if constraint is not None:
        processor = _build_constraint_logits_processor(model, tokenizer, constraint)
        try:
            logits_for_sampling = _apply_constraint_logits_for_context(
                logits=logits_temp,
                processor=processor,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to apply constrained sampling ({constraint.kind}): {e}",
            )
        valid_mask = torch.isfinite(logits_for_sampling)
        if int(valid_mask.sum().item()) <= 0:
            raise HTTPException(
                status_code=400,
                detail="Constraint has no valid next token at this position.",
            )

    probs_sampling = torch.softmax(logits_for_sampling, dim=-1)
    if not torch.isfinite(probs_sampling).all():
        raise HTTPException(
            status_code=400,
            detail="Sampling distribution became invalid for the current settings.",
        )
    sorted_probs_sampling, sorted_ids_sampling = torch.sort(probs_sampling, descending=True)

    # Determine nucleus cutoff count (keep at least one token).
    if top_p >= 1.0:
        kept_count = int(sorted_probs_sampling.numel())
    else:
        cum = torch.cumsum(sorted_probs_sampling, dim=-1)
        mask = cum <= top_p
        if mask.numel() > 0:
            mask[0] = True
        kept_count = int(mask.sum().item())

    kept_ids = sorted_ids_sampling[:kept_count]
    kept_probs = sorted_probs_sampling[:kept_count]
    denom = kept_probs.sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        raise HTTPException(
            status_code=400,
            detail="No valid probability mass after applying current settings.",
        )
    kept_probs_norm = kept_probs / denom

    k = min(top_k, int(sorted_ids_display.numel()))
    top_ids = sorted_ids_display[:k]
    top_base_probs = probs_base[top_ids]
    top_ids_list = [int(x) for x in top_ids.detach().cpu().tolist()]
    top_tokens = _token_pills_from_ids(tokenizer, top_ids_list)

    kept_id_list = [int(x) for x in kept_ids.detach().cpu().tolist()]
    kept_prob_list = [float(x) for x in kept_probs_norm.detach().cpu().tolist()]
    kept_prob_map = {tid: prob for tid, prob in zip(kept_id_list, kept_prob_list)}

    top_final_probs: List[float] = []
    for i, tok in enumerate(top_tokens):
        tid = top_ids_list[i]
        p_final = float(kept_prob_map.get(tid, 0.0))
        is_valid = True
        if valid_mask is not None:
            is_valid = bool(valid_mask[tid].detach().cpu().item())
            if not is_valid:
                p_final = 0.0
        tok["p_base"] = float(top_base_probs[i].detach().cpu().item())
        tok["p_final"] = p_final
        tok["kept"] = bool(tid in kept_prob_map and is_valid)
        tok["valid"] = bool(is_valid)
        top_final_probs.append(p_final)

    other_final = float(max(0.0, 1.0 - float(sum(top_final_probs))))
    other_base = float(max(0.0, 1.0 - float(top_base_probs.sum().item())))

    dist = {
        "top_k": top_k,
        "tokens": top_tokens,
        "other_p_base": other_base,
        "other_p_final": other_final,
        "kept_token_count": int(kept_count),
        "constraint_active": bool(constraint is not None),
        "constraint_type": constraint.kind if constraint is not None else None,
    }

    return {
        "dist": dist,
        "kept_ids": kept_ids.detach().cpu(),
        "kept_probs": kept_probs_norm.detach().cpu(),
        "valid_mask": valid_mask.detach().cpu() if valid_mask is not None else None,
    }


def _compute_next_token_distribution(
    model,
    tokenizer,
    prompt_ids: List[int],
    completion_ids: List[int],
    top_k: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    constraint: Optional[ConstraintSpec] = None,
) -> Dict[str, Any]:
    sampling_state = _compute_sampling_state(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        top_k=top_k,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        constraint=constraint,
    )
    return sampling_state["dist"]


def _is_token_allowed(valid_mask: Optional[torch.Tensor], token_id: int) -> bool:
    if valid_mask is None:
        return True
    tid = int(token_id)
    if tid < 0 or tid >= int(valid_mask.numel()):
        return False
    return bool(valid_mask[tid].item())


class DecodingParams(BaseModel):
    greedy: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)


class ConstraintConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    enabled: bool = False
    type: str = "multiple_choice"
    schema_text: str = Field(default="", alias="schema")


class PreviewRequest(BaseModel):
    branch_id: str = "default"
    prompt: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat_template: Optional[bool] = None
    completion_ids: List[int] = Field(default_factory=list)
    top_k: int = 10
    params: DecodingParams = Field(default_factory=DecodingParams)
    constraint: ConstraintConfig = Field(default_factory=ConstraintConfig)


class StepRequest(BaseModel):
    branch_id: str = "default"
    reset_rng: bool = False
    prompt: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat_template: Optional[bool] = None
    completion_ids: List[int] = Field(default_factory=list)
    top_k: int = 10
    params: DecodingParams = Field(default_factory=DecodingParams)
    constraint: ConstraintConfig = Field(default_factory=ConstraintConfig)
    force_token_id: Optional[int] = None
    force_text: Optional[str] = None


class AttributionRequest(BaseModel):
    branch_id: str = "default"
    prompt: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat_template: Optional[bool] = None
    completion_ids: List[int] = Field(default_factory=list)
    target_index: int = 0
    target_token_id: Optional[int] = None
    method: str = "off"


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
    constraint_spec = _resolve_constraint_spec(req.constraint)

    prompt_ids, full_ids = _build_input_ids(
        tok,
        req.prompt,
        req.completion_ids,
        use_chat_template=req.use_chat_template,
        system_prompt=req.system_prompt,
    )

    dist = _compute_next_token_distribution(
        model=model,
        tokenizer=tok,
        prompt_ids=prompt_ids,
        completion_ids=[int(x) for x in (req.completion_ids or [])],
        top_k=req.top_k,
        temperature=req.params.temperature,
        top_p=req.params.top_p,
        repetition_penalty=req.params.repetition_penalty,
        presence_penalty=req.params.presence_penalty,
        frequency_penalty=req.params.frequency_penalty,
        constraint=constraint_spec,
    )

    # Display tokens exactly as the model sees them (including chat template tokens).
    prompt_tokens = _token_pills_from_ids(tok, prompt_ids)
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
    constraint_spec = _resolve_constraint_spec(req.constraint)

    if constraint_spec is not None and req.force_text is not None and req.force_text != "":
        raise HTTPException(
            status_code=400,
            detail="Force text is disabled while constrained sampling is active.",
        )

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

        # Compute distribution for each forced token at the time it was appended.
        appended_meta: List[Dict[str, Any]] = []
        current_completion = [int(x) for x in (req.completion_ids or [])]
        for fid in forced_ids:
            prompt_ids, _full_ids = _build_input_ids(
                tok,
                req.prompt,
                current_completion,
                use_chat_template=req.use_chat_template,
                system_prompt=req.system_prompt,
            )
            dist = _compute_next_token_distribution(
                model=model,
                tokenizer=tok,
                prompt_ids=prompt_ids,
                completion_ids=current_completion,
                top_k=req.top_k,
                temperature=req.params.temperature,
                top_p=req.params.top_p,
                repetition_penalty=req.params.repetition_penalty,
                presence_penalty=req.params.presence_penalty,
                frequency_penalty=req.params.frequency_penalty,
                constraint=constraint_spec,
            )
            appended_meta.append(
                {
                    "dist": dist,
                    "selected_id": int(fid),
                    "forced": True,
                    "forced_by": "text",
                }
            )
            current_completion.append(int(fid))

        appended_tokens = _token_pills_from_ids(tok, forced_ids)
        dist = appended_meta[0]["dist"] if appended_meta else None

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
            "appended_meta": appended_meta,
            "new_completion_ids": new_completion,
            "done": done,
            "stop_reason": stop_reason,
        }

    # Normal step: compute distribution, then choose token (greedy / sampled / forced token id)
    prompt_ids, _full_ids = _build_input_ids(
        tok,
        req.prompt,
        req.completion_ids,
        use_chat_template=req.use_chat_template,
        system_prompt=req.system_prompt,
    )

    sampling_state = _compute_sampling_state(
        model=model,
        tokenizer=tok,
        prompt_ids=prompt_ids,
        completion_ids=[int(x) for x in (req.completion_ids or [])],
        top_k=req.top_k,
        temperature=req.params.temperature,
        top_p=req.params.top_p,
        repetition_penalty=req.params.repetition_penalty,
        presence_penalty=req.params.presence_penalty,
        frequency_penalty=req.params.frequency_penalty,
        constraint=constraint_spec,
    )
    dist = sampling_state["dist"]
    kept_ids_cpu = sampling_state["kept_ids"]
    kept_probs_cpu = sampling_state["kept_probs"]
    valid_mask_cpu = sampling_state["valid_mask"]

    # Choose token
    selected_id: int
    selected_prob_final: Optional[float] = None

    if req.force_token_id is not None:
        selected_id = int(req.force_token_id)
        if constraint_spec is not None and not _is_token_allowed(valid_mask_cpu, selected_id):
            raise HTTPException(
                status_code=400,
                detail="Forced token is invalid under the active constrained sampling schema.",
            )
    else:
        if int(kept_ids_cpu.numel()) <= 0:
            raise HTTPException(status_code=400, detail="No token available to sample.")
        if bool(req.params.greedy):
            selected_id = int(kept_ids_cpu[0].item())
            selected_prob_final = float(kept_probs_cpu[0].item())
        else:
            gen = _get_rng(req.branch_id, req.params.seed, reset=bool(req.reset_rng))
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
        "appended_meta": [
            {
                "dist": dist,
                "selected_id": int(selected_id),
                "forced": bool(req.force_token_id is not None),
                "forced_by": "token" if req.force_token_id is not None else None,
            }
        ],
        "new_completion_ids": new_completion,
        "done": done,
        "stop_reason": stop_reason,
    }


@app.post("/api/attribution")
def attribution(req: AttributionRequest):
    method = (req.method or "").strip().lower()
    if method in ("", "off"):
        return {"ok": True, "attributions": [], "source_len": 0}
    if method not in {"attention", "saliency", "input_x_gradient", "integrated_gradients"}:
        raise HTTPException(status_code=400, detail="Unsupported attribution method.")

    attr_model, tok, public = _get_attr_model(method)
    if tok.pad_token_id is None:
        raise HTTPException(status_code=500, detail="Tokenizer has no pad token; attribution requires padding.")
    attr_tok = getattr(attr_model, "tokenizer", None)
    if attr_tok is not None and attr_tok.pad_token_id is None:
        raise HTTPException(status_code=500, detail="Attribution tokenizer has no pad token; attribution requires padding.")

    if method == "attention":
        attn_model = getattr(attr_model, "model", None) or getattr(attr_model, "model_ref", None)
        if attn_model is not None and hasattr(attn_model, "set_attn_implementation"):
            try:
                attn_model.set_attn_implementation("eager")
            except Exception:
                pass
    if method == "integrated_gradients" and public.device == "mps":
        _patch_captum_ig_mps()

    completion_ids = [int(x) for x in (req.completion_ids or [])]
    target_index = int(req.target_index or 0)
    if target_index < 0:
        target_index = 0
    target_token_id = req.target_token_id
    if target_token_id is None:
        if target_index >= len(completion_ids):
            raise HTTPException(status_code=400, detail="target_index out of range.")
        target_token_id = completion_ids[target_index]

    target_ids = [int(target_token_id)]
    context_ids = completion_ids[:target_index]

    # Build context text from tokens to preserve tokenization alignment.
    prompt_ids, _ = _build_input_ids(
        tok,
        req.prompt,
        context_ids,
        use_chat_template=req.use_chat_template,
        system_prompt=req.system_prompt,
    )
    context_full_ids = [int(x) for x in (prompt_ids + context_ids)]
    context_tokens = _ids_to_tokens(tok, context_full_ids)
    target_tokens = _ids_to_tokens(tok, target_ids)
    context_text = _tokens_to_string(tok, context_tokens)
    full_generated_text = _tokens_to_string(tok, context_tokens + target_tokens)

    gen_ids = tok(full_generated_text, add_special_tokens=True, return_tensors=None)["input_ids"]
    if isinstance(gen_ids, (list, tuple)) and len(gen_ids) > 0 and isinstance(gen_ids[0], (list, tuple)):
        gen_ids = gen_ids[0]
    if isinstance(gen_ids, torch.Tensor):
        gen_ids = gen_ids.detach().cpu().tolist()
    if isinstance(gen_ids, np.ndarray):
        gen_ids = gen_ids.tolist()
    gen_ids = [int(x) for x in list(gen_ids or [])]
    prefix_len = len(context_tokens)
    if len(gen_ids) <= prefix_len:
        prefix_len = max(len(gen_ids) - 1, 0)
    def _run_attr(model_for_attr):
        prev_dtype = None
        try:
            if method == "integrated_gradients":
                prev_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
            out = model_for_attr.attribute(
                input_texts=context_text,
                generated_texts=full_generated_text,
                method=method,
                attr_pos_start=prefix_len,
                attr_pos_end=prefix_len + 1,
                show_progress=False,
                pretty_progress=False,
            )
            agg = out.aggregate()
            seq = agg.sequence_attributions[0]
            scores = seq.source_attributions
            if scores is None or (hasattr(scores, "numel") and scores.numel() == 0):
                scores = seq.target_attributions
            if scores is None:
                return None
            if isinstance(scores, torch.Tensor):
                while scores.dim() > 2:
                    scores = scores.sum(dim=-1)
                if scores.dim() == 2:
                    scores = scores[:, 0] if scores.shape[1] > 0 else scores.reshape(-1)
                return scores.detach().cpu().tolist()
            return list(scores)
        finally:
            if prev_dtype is not None:
                torch.set_default_dtype(prev_dtype)

    used_cpu = False
    try:
        vals = _run_attr(attr_model)
    except Exception as e:
        msg = str(e)
        attention_err = (
            ("attention" in msg.lower() and "not support" in msg.lower())
            or ("output_attentions" in msg.lower())
            or ("no scores" in msg.lower())
        )
        if method == "attention" and public.device != "cpu" and attention_err:
            logger.warning("Attention not available on %s, retrying on CPU.", public.device)
            cpu_attr_model, _cpu_tok = _get_attr_model_cpu(public.model_id)
            try:
                vals = _run_attr(cpu_attr_model)
                used_cpu = True
                if vals is None or len(vals) == 0:
                    return {"ok": False, "error": "attention_no_scores", "attributions": [], "source_len": 0}
            except Exception as e2:
                logger.exception(
                    "Attribution failed on CPU fallback",
                    extra={
                        "method": method,
                        "model_id": public.model_id,
                        "target_index": target_index,
                        "target_token_id": target_token_id,
                        "context_len": len(context_full_ids),
                        "target_len": len(target_ids),
                        "device": "cpu",
                    },
                )
                raise HTTPException(status_code=500, detail=f"Attribution failed: {e2}")
        else:
            if method == "attention" and attention_err:
                logger.warning("Attention attribution not supported: %s", msg)
                return {"ok": False, "error": "attention_not_supported", "attributions": [], "source_len": 0}
            logger.exception(
                "Attribution failed",
                extra={
                    "method": method,
                    "model_id": public.model_id,
                    "target_index": target_index,
                    "target_token_id": target_token_id,
                    "context_len": len(context_full_ids),
                    "target_len": len(target_ids),
                },
            )
            raise HTTPException(status_code=500, detail=f"Attribution failed: {e}")

    if method == "attention" and public.device != "cpu" and not used_cpu:
        max_abs = 0.0
        if vals:
            for v in vals:
                try:
                    av = abs(float(v))
                except Exception:
                    av = 0.0
                if av > max_abs:
                    max_abs = av
        if vals is None or len(vals) == 0 or max_abs == 0.0:
            logger.warning("Attention produced empty/zero scores on %s, retrying on CPU.", public.device)
            cpu_attr_model, _cpu_tok = _get_attr_model_cpu(public.model_id)
            vals = _run_attr(cpu_attr_model)
            if vals is None or len(vals) == 0:
                return {"ok": False, "error": "attention_no_scores", "attributions": [], "source_len": 0}

    return {
        "ok": True,
        "attributions": _sanitize_floats(vals or []),
        "source_len": int(len(vals or [])),
        "context_len": int(len(context_full_ids)),
        "target_id": int(target_token_id),
        "model_id": public.model_id,
        "method": method,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level=LOG_LEVEL)
