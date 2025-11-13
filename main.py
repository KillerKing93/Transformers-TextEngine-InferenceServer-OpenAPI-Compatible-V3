#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI Inference Server (OpenAI-compatible) for Qwen3-VL multimodal model.

- Default model: unsloth/Qwen3-4B-Instruct-2507
- Endpoints:
  * GET /openapi.yaml     (OpenAPI schema in YAML)
  * GET /health           (readiness + context report)
  * POST /v1/chat/completions (non-stream and streaming SSE)
  * POST /v1/cancel/{session_id} (custom cancel endpoint)

Notes:
- Uses Hugging Face Transformers with trust_remote_code=True.
- Supports OpenAI-style chat messages with text, image_url/input_image, video_url/input_video.
- Streaming SSE supports resume (session_id + Last-Event-ID) with optional SQLite persistence.
- Auto prompt compression prevents context overflow with a simple truncate strategy.
"""

import os
import io
import re
import base64
import tempfile
import contextlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Deque, Literal

from fastapi import FastAPI, HTTPException, Request, Header, Query, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse
from fastapi.responses import StreamingResponse, Response, FileResponse
from starlette.staticfiles import StaticFiles
import json
import yaml
import threading
import time
import uuid
import sqlite3
from collections import deque
import subprocess
import sys
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure HF cache dirs are relative to this project by default
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HF_CACHE = os.path.join(ROOT_DIR, "hf-cache")
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = DEFAULT_HF_CACHE
# Remove deprecated TRANSFORMERS_CACHE to avoid warnings
if os.getenv("TRANSFORMERS_CACHE"):
    del os.environ["TRANSFORMERS_CACHE"]
# Create directory eagerly to avoid later mkdir races
try:
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
except Exception:
    pass

# Optional heavy deps are imported lazily inside Engine to improve startup UX
import requests
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download, get_hf_file_metadata

# OCR import (Disabled - use image captioning instead)
# try:
#     from rapidocr_onnxruntime import RapidOCR
# except ImportError:
#     RapidOCR = None
RapidOCR = None  # Disabled

# Server config
PORT = int(os.getenv("PORT", "3000"))
DEFAULT_MODEL_ID = os.getenv("MODEL_REPO_ID", "unsloth/Qwen3-4B-Instruct-2507")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
# Default max tokens: honor env, fallback to 4096 as previously discussed
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "16"))
DEVICE_MAP = os.getenv("DEVICE_MAP", "cpu")  # Force CPU for current deployment
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float32")  # float32 is faster on CPU

# Quantization config (BitsAndBytes) - disabled for CPU deployment
LOAD_IN_4BIT = str(os.getenv("LOAD_IN_4BIT", "0")).lower() in ("1", "true", "yes", "y")  # Disabled
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16")
BNB_4BIT_USE_DOUBLE_QUANT = str(os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "1")).lower() in ("1", "true", "yes", "y")
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")

# Concurrency config
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # Thread pool for concurrent processing
OCR_TIMEOUT_SECONDS = int(os.getenv("OCR_TIMEOUT_SECONDS", "120"))  # 2 minute timeout for OCR

# Persistent session store (SQLite)
PERSIST_SESSIONS = str(os.getenv("PERSIST_SESSIONS", "0")).lower() in ("1", "true", "yes", "y")
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "sessions.db")
SESSIONS_TTL_SECONDS = int(os.getenv("SESSIONS_TTL_SECONDS", "600"))
# Auto-cancel if all clients disconnect for duration (seconds). 0 disables it.
CANCEL_AFTER_DISCONNECT_SECONDS = int(os.getenv("CANCEL_AFTER_DISCONNECT_SECONDS", "3600"))

# Auto compression settings
ENABLE_AUTO_COMPRESSION = str(os.getenv("ENABLE_AUTO_COMPRESSION", "1")).lower() in ("1", "true", "yes", "y")
CONTEXT_MAX_TOKENS_AUTO = int(os.getenv("CONTEXT_MAX_TOKENS_AUTO", "0"))  # 0 -> infer from model/tokenizer
CONTEXT_SAFETY_MARGIN = int(os.getenv("CONTEXT_SAFETY_MARGIN", "256"))
COMPRESSION_STRATEGY = os.getenv("COMPRESSION_STRATEGY", "truncate")  # truncate | summarize (future)

# Eager model loading (download/check at startup before serving traffic)
EAGER_LOAD_MODEL = str(os.getenv("EAGER_LOAD_MODEL", "1")).lower() in ("1", "true", "yes", "y")

# Global thread pool executor for concurrent processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="inference")

# Global OCR engine (Disabled)
# _ocr_engine = None

# def get_ocr_engine():
#     global _ocr_engine
#     if _ocr_engine is None and RapidOCR is not None:
#         try:
#             _ocr_engine = RapidOCR()
#             print("[OCR] RapidOCR engine initialized")
#         except Exception as e:
#             print(f"[OCR] Failed to initialize RapidOCR: {e}")
#             _ocr_engine = None
#     return _ocr_engine
def get_ocr_engine():
    """Placeholder function - OCR disabled"""
    return None

def _log(msg: str):
    # Consistent, flush-immediate startup logs
    print(f"[startup] {msg}", flush=True)

def prefetch_model_assets(repo_id: str, token: Optional[str]) -> Optional[str]:
    """
    Reproducible prefetch driven by huggingface-cli:
    - Downloads the ENTIRE repo using CLI (visible progress bar).
    - Returns the local directory path where the repo is mirrored.
    - If CLI is unavailable, falls back to verbose API prefetch.
    """
    try:
        # Enable accelerated transfer only if hf_transfer is installed; otherwise disable to avoid runtime errors on Spaces
        try:
            import importlib.util as _imputil
            if _imputil.find_spec("hf_transfer") is not None:
                os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            else:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        except Exception:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        # XET acceleration if available; harmless if missing
        os.environ.setdefault("HF_HUB_ENABLE_XET", "1")

        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or ""
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Resolve huggingface-cli path (Windows-friendly) - try hf first, fallback to huggingface-cli
        cli_path = shutil.which("hf")
        if not cli_path:
            cli_path = shutil.which("huggingface-cli")
        if not cli_path:
            candidates = []
            appdata = os.getenv("APPDATA")
            if appdata:
                candidates.append(os.path.join(appdata, "Python", "Python312", "Scripts", "hf.exe"))
                candidates.append(os.path.join(appdata, "Python", "Python312", "Scripts", "huggingface-cli.exe"))
            candidates.append(os.path.join(os.path.dirname(sys.executable), "Scripts", "hf.exe"))
            candidates.append(os.path.join(os.path.dirname(sys.executable), "Scripts", "huggingface-cli.exe"))
            cli_path = next((p for p in candidates if os.path.exists(p)), None)

        # Preferred: one-shot CLI download for the whole repo (shows live progress)
        if cli_path:
            local_root = os.path.join(cache_dir if cache_dir else ".", repo_id.replace("/", "_"))
            os.makedirs(local_root, exist_ok=True)
            _log(f"Using hf download to download entire repo -> '{local_root}'")
            cmd = [
                cli_path,
                "download",
                "--repo-type",
                "model",
                "--local-dir",
                local_root,
                repo_id,
            ]
            if token:
                cmd += ["--token", token]
            # Inherit stdio; users will see a proper progress bar
            subprocess.run(cmd, check=False)
            # Verify we have the essential files
            if os.path.exists(os.path.join(local_root, "config.json")) or os.path.exists(os.path.join(local_root, "model.safetensors")):
                _log("CLI prefetch completed")
                return local_root
            else:
                _log("CLI prefetch finished but essential files not found; will fallback to API mirroring")

        # Fallback: verbose API-driven prefetch with per-file logging
        _log(f"Prefetching (API) repo={repo_id} to cache='{cache_dir}'")
        try:
            files = list_repo_files(repo_id, repo_type="model", token=token)
        except Exception as e:
            _log(f"list_repo_files failed ({type(e).__name__}: {e}); falling back to snapshot_download")
            snapshot_download(repo_id, token=token, local_files_only=False)
            _log("Prefetch completed (snapshot)")
            return None

        total = len(files)
        _log(f"Found {total} files to ensure cached (API)")
        for i, fn in enumerate(files, start=1):
            try:
                meta = get_hf_file_metadata(repo_id, fn, repo_type="model", token=token)
                size_bytes = meta.size or 0
            except Exception:
                size_bytes = 0
            size_mb = size_bytes / (1024 * 1024) if size_bytes else 0.0
            _log(f"[{i}/{total}] fetching '{fn}' (~{size_mb:.2f} MB)")
            _ = hf_hub_download(
                repo_id=repo_id,
                filename=fn,
                repo_type="model",
                token=token,
                local_files_only=False,
                resume_download=True,
            )
            _log(f"[{i}/{total}] done '{fn}'")
        _log("Prefetch completed (API)")
        return None
    except Exception as e:
        _log(f"Prefetch skipped: {type(e).__name__}: {e}")
        return None

def is_data_url(url: str) -> bool:
    return url.startswith("data:") and ";base64," in url


def is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def decode_base64_to_bytes(b64: str) -> bytes:
    # strip possible "data:*;base64," prefix
    if "base64," in b64:
        b64 = b64.split("base64,", 1)[1]
    return base64.b64decode(b64, validate=False)


def fetch_bytes(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> bytes:
    if not is_http_url(url):
        raise ValueError(f"Only http(s) URLs supported for fetch, got: {url}")
    resp = requests.get(url, headers=headers or {}, timeout=timeout, stream=True)
    resp.raise_for_status()
    return resp.content


def load_image_from_any(src: Dict[str, Any]) -> Image.Image:
    """
    src can be:
      - { "url": "http(s)://..." } (also supports data URL)
      - { "b64_json": "<base64>" }
      - { "path": "local_path" } (optional)
    """
    if "b64_json" in src and src["b64_json"]:
        data = decode_base64_to_bytes(str(src["b64_json"]))
        return Image.open(io.BytesIO(data)).convert("RGB")

    if "url" in src and src["url"]:
        url = str(src["url"])
        if is_data_url(url):
            data = decode_base64_to_bytes(url)
            return Image.open(io.BytesIO(data)).convert("RGB")
        if is_http_url(url):
            data = fetch_bytes(url)
            return Image.open(io.BytesIO(data)).convert("RGB")
        # treat as local path
        if os.path.exists(url):
            with open(url, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        raise ValueError(f"Invalid image url/path: {url}")

    if "path" in src and src["path"]:
        p = str(src["path"])
        if os.path.exists(p):
            with open(p, "rb") as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        raise ValueError(f"Image path not found: {p}")

    raise ValueError("Unsupported image source payload")


def write_bytes_tempfile(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        f.write(data)
    return tmp.name


def load_video_frames_from_any(src: Dict[str, Any], max_frames: int = MAX_VIDEO_FRAMES) -> List[Image.Image]:
    """
    Returns a list of PIL.Image frames (RGB) sampled up to max_frames.
    src can be:
      - { "url": "http(s)://..." } (mp4/mov/webm/etc.)
      - { "b64_json": "<base64 of a video file>" }
      - { "path": "local_path" }
    """
    # Prefer imageio.v3 if present, fallback to OpenCV
    # We load all frames then uniform sample if too many.
    def _load_all_frames(path: str) -> List[Image.Image]:
        frames: List[Image.Image] = []
        with contextlib.suppress(ImportError):
            import imageio.v3 as iio
            arr_iter = iio.imiter(path)  # yields numpy arrays HxWxC
            for arr in arr_iter:
                if arr is None:
                    continue
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                frames.append(Image.fromarray(arr).convert("RGB"))
            return frames

        # Fallback to OpenCV
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        while ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            ok, frame = cap.read()
        cap.release()
        return frames

    # Resolve to a local path
    local_path = None
    if "b64_json" in src and src["b64_json"]:
        data = decode_base64_to_bytes(str(src["b64_json"]))
        local_path = write_bytes_tempfile(data, suffix=".mp4")
    elif "url" in src and src["url"]:
        url = str(src["url"])
        if is_data_url(url):
            data = decode_base64_to_bytes(url)
            local_path = write_bytes_tempfile(data, suffix=".mp4")
        elif is_http_url(url):
            data = fetch_bytes(url)
            local_path = write_bytes_tempfile(data, suffix=".mp4")
        elif os.path.exists(url):
            local_path = url
        else:
            raise ValueError(f"Invalid video url/path: {url}")
    elif "path" in src and src["path"]:
        p = str(src["path"])
        if os.path.exists(p):
            local_path = p
        else:
            raise ValueError(f"Video path not found: {p}")
    else:
        raise ValueError("Unsupported video source payload")

    frames = _load_all_frames(local_path)
    # Uniform sample if too many frames
    if len(frames) > max_frames and max_frames > 0:
        idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int).tolist()
        frames = [frames[i] for i in idxs]
    return frames


class ChatRequest(BaseModel):
    """OpenAI-compatible Chat Completions request body."""
    model: Optional[str] = Field(default=None, description="Model id (defaults to env MODEL_REPO_ID).")
    messages: List[Dict[str, Any]] = Field(description="OpenAI-style messages array. Supports text, image_url/input_image, video_url/input_video parts.")
    max_tokens: Optional[int] = Field(default=None, description="Max new tokens to generate.")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature.")
    stream: Optional[bool] = Field(default=None, description="When true, returns Server-Sent Events stream.")
    session_id: Optional[str] = Field(default=None, description="Optional session id for resumable SSE.")
    # Pydantic v2 schema extras with rich examples
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Text-only",
                    "value": {
                        "messages": [
                            {"role": "user", "content": "Hello, summarize the benefits of multimodal LLMs."}
                        ],
                        "max_tokens": 128
                    }
                },
                {
                    "summary": "Image by URL",
                    "value": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What is in this image?"},
                                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                                ]
                            }
                        ],
                        "max_tokens": 128
                    }
                },
                {
                    "summary": "Video by URL (streaming SSE)",
                    "value": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this clip briefly."},
                                    {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
                                ]
                            }
                        ],
                        "stream": True,
                        "max_tokens": 128
                    }
                }
            ]
        }
    )

class MessageModel(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChoiceModel(BaseModel):
    index: int
    message: MessageModel
    finish_reason: Optional[str] = None

class UsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """Non-streaming Chat Completions response (when stream=false)."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChoiceModel]
    usage: UsageModel
    context: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    ok: bool
    modelReady: bool
    modelId: str
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class CancelResponse(BaseModel):
    ok: bool
    session_id: str


class Engine:
    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        # Lazy import heavy deps
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModel, BitsAndBytesConfig
        # AutoModelForImageTextToText is the v5+ replacement for Vision2Seq in Transformers
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore
        except Exception:
            AutoModelForImageTextToText = None  # type: ignore

        # Resolve device map to avoid 'meta' device on CPU Spaces
        # If DEVICE_MAP is "auto" but no CUDA is available, force "cpu" and disable low_cpu_mem_usage
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if hf_token:
            # Only pass 'token' (use_auth_token is deprecated and causes conflicts)
            model_kwargs["token"] = hf_token

        # Add quantization config if enabled
        if LOAD_IN_4BIT:
            try:
                import torch
                compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE, torch.float16)
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
                    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
                )
                model_kwargs["quantization_config"] = quant_config
                _log(f"Using 4-bit quantization: {BNB_4BIT_QUANT_TYPE}, compute_dtype={BNB_4BIT_COMPUTE_DTYPE}, double_quant={BNB_4BIT_USE_DOUBLE_QUANT}")
            except Exception as e:
                _log(f"BitsAndBytes quantization failed: {e}; falling back to full precision")

        # Device and dtype resolution
        try:
            import torch  # local import to avoid heavy import at module load
            has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            has_cuda = False

        resolved_device_map = DEVICE_MAP
        if str(DEVICE_MAP).lower() == "auto" and not has_cuda:
            resolved_device_map = "cpu"

        model_kwargs["device_map"] = resolved_device_map
        # Explicitly disable low_cpu_mem_usage on pure CPU to fully materialize weights (avoids meta tensors)
        if resolved_device_map == "cpu":
            model_kwargs["low_cpu_mem_usage"] = False
        # dtype - use 'dtype' instead of deprecated 'torch_dtype'
        if TORCH_DTYPE != "auto":
            try:
                import torch
                model_kwargs["dtype"] = getattr(torch, TORCH_DTYPE, TORCH_DTYPE)
            except Exception:
                model_kwargs["dtype"] = TORCH_DTYPE
        else:
            model_kwargs["dtype"] = "auto"
        # store for later
        self._resolved_device_map = resolved_device_map

        # Processor (handles text + images/videos)
        proc_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if hf_token:
            proc_kwargs["token"] = hf_token
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            **proc_kwargs,
        )  # pragma: no cover

        # Prefer ImageTextToText (Transformers v5 path), then Vision2Seq, then CausalLM as a last resort
        model = None
        if 'AutoModelForImageTextToText' in globals() and AutoModelForImageTextToText is not None:
            try:
                model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)  # pragma: no cover
            except Exception:
                model = None
        if model is None:
            try:
                # AutoModelForVision2Seq is deprecated, but try it for compatibility
                model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)  # pragma: no cover
            except Exception:
                model = None
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)  # pragma: no cover
            except Exception:
                model = None
        if model is None:
            # Generic AutoModel as last-resort with trust_remote_code to load custom architectures
            model = AutoModel.from_pretrained(model_id, **model_kwargs)  # pragma: no cover
        self.model = model.eval()  # pragma: no cover
        # Ensure model is fully on CPU when resolved device_map is cpu (prevents meta device mix during inference)
        try:
            if str(getattr(self, "_resolved_device_map", "")).lower() == "cpu":
                _ = self.model.to("cpu")
        except Exception:
            pass
        # Ensure model is on CPU when resolved device_map is cpu (prevents meta device mix during inference)
        try:
            if getattr(self, "_resolved_device_map", None) == "cpu":
                _ = self.model.to("cpu")
        except Exception:
            pass

        self.model_id = model_id
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.last_context_info: Dict[str, Any] = {}

    def _model_max_context(self) -> int:
        try:
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                v = getattr(cfg, "max_position_embeddings", None)
                if isinstance(v, int) and v > 0 and v < 10_000_000:
                    return v
        except Exception:
            pass
        try:
            mx = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
            if mx > 0 and mx < 10_000_000_000:
                return mx
        except Exception:
            pass
        return 32768

    def _count_prompt_tokens(self, text: str) -> int:
        try:
            if self.tokenizer is not None:
                enc = self.tokenizer([text], add_special_tokens=False, return_attention_mask=False)
                ids = enc["input_ids"][0]
                return len(ids)
        except Exception:
            pass
        return max(1, int(len(text.split()) * 1.3))

    def _auto_compress_if_needed(
        self, mm_messages: List[Dict[str, Any]], max_new_tokens: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {}
        # Build once to measure
        text0 = self.processor.apply_chat_template(mm_messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = self._count_prompt_tokens(text0)
        max_ctx = CONTEXT_MAX_TOKENS_AUTO if CONTEXT_MAX_TOKENS_AUTO > 0 else self._model_max_context()
        budget = max(1024, max_ctx - CONTEXT_SAFETY_MARGIN - int(max_new_tokens))
        if not ENABLE_AUTO_COMPRESSION or prompt_tokens <= budget:
            info = {
                "compressed": False,
                "prompt_tokens": int(prompt_tokens),
                "max_context": int(max_ctx),
                "budget": int(budget),
                "strategy": COMPRESSION_STRATEGY,
                "dropped_messages": 0,
            }
            return mm_messages, info

        # Truncate earliest non-system messages until within budget
        msgs = list(mm_messages)
        dropped = 0
        guard = 0
        while True:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_tokens = self._count_prompt_tokens(text)
            if prompt_tokens <= budget or len(msgs) <= 1:
                break
            # drop earliest non-system
            drop_idx = None
            for j, m in enumerate(msgs):
                if (m.get("role") or "user") != "system":
                    drop_idx = j
                    break
            if drop_idx is None:
                break
            msgs.pop(drop_idx)
            dropped += 1
            guard += 1
            if guard > 10000:
                break

        info = {
            "compressed": True,
            "prompt_tokens": int(prompt_tokens),
            "max_context": int(max_ctx),
            "budget": int(budget),
            "strategy": "truncate",
            "dropped_messages": int(dropped),
        }
        return msgs, info

    def get_context_report(self) -> Dict[str, Any]:
        try:
            tk_max = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        except Exception:
            tk_max = 0
        return {
            "compressionEnabled": ENABLE_AUTO_COMPRESSION,
            "strategy": COMPRESSION_STRATEGY,
            "safetyMargin": CONTEXT_SAFETY_MARGIN,
            "modelMaxContext": self._model_max_context(),
            "tokenizerModelMaxLength": tk_max,
            "last": self.last_context_info or {},
        }

    def build_mm_messages(
        self, openai_messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[List[Image.Image]]]:
        """
        Convert OpenAI-style messages to Qwen multimodal messages.
        Returns:
          - messages for apply_chat_template
          - flat list of images in encounter order
          - list of videos (each is list of PIL frames)
        """
        mm_msgs: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        videos: List[List[Image.Image]] = []

        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            parts: List[Dict[str, Any]] = []

            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    ptype = p.get("type")
                    if ptype == "text":
                        txt = p.get("text", "")
                        if txt:
                            parts.append({"type": "text", "text": txt})
                    elif ptype in ("image_url", "input_image"):
                        src: Dict[str, Any] = {}
                        if ptype == "image_url":
                            u = (p.get("image_url") or {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
                            src["url"] = u
                        else:
                            b64 = p.get("image") or p.get("b64_json") or p.get("data") or (p.get("image_url") or {}).get("url")
                            if b64:
                                src["b64_json"] = b64
                        try:
                            img = load_image_from_any(src)
                            images.append(img)
                            parts.append({"type": "image", "image": img})
                        except Exception as e:
                            raise ValueError(f"Failed to parse image part: {e}") from e
                    elif ptype in ("video_url", "input_video"):
                        src = {}
                        if ptype == "video_url":
                            u = (p.get("video_url") or {}).get("url") if isinstance(p.get("video_url"), dict) else p.get("video_url")
                            src["url"] = u
                        else:
                            b64 = p.get("video") or p.get("b64_json") or p.get("data")
                            if b64:
                                src["b64_json"] = b64
                        try:
                            frames = load_video_frames_from_any(src, max_frames=MAX_VIDEO_FRAMES)
                            videos.append(frames)
                            parts.append({"type": "video", "video": frames})
                        except Exception as e:
                            raise ValueError(f"Failed to parse video part: {e}") from e
                    else:
                        if isinstance(p, dict):
                            txt = p.get("text")
                            if isinstance(txt, str) and txt:
                                parts.append({"type": "text", "text": txt})
            else:
                if content:
                    parts.append({"type": "text", "text": str(content)})

            mm_msgs.append({"role": role, "content": parts})

        return mm_msgs, images, videos

    def infer(self, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        mm_messages, images, videos = self.build_mm_messages(messages)
        # Auto-compress if needed based on context budget
        mm_messages, ctx_info = self._auto_compress_if_needed(mm_messages, max_tokens)
        self.last_context_info = ctx_info

        # Build chat template
        text = self.processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        proc_kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos

        inputs = self.processor(**proc_kwargs)
        # Move tensors to the correct device
        try:
            if str(getattr(self, "_resolved_device_map", "")).lower() == "cpu":
                # Explicit CPU placement avoids 'meta' device errors on Spaces
                inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}
            else:
                device = getattr(self.model, "device", None) or next(self.model.parameters()).device
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass

        do_sample = temperature is not None and float(temperature) > 0.0

        # Safer on CPU: run without gradients to reduce memory pressure and avoid autograd hooks
        try:
            import torch
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_tokens),
                    temperature=float(temperature),
                    do_sample=do_sample,
                    use_cache=True,
                )
        except Exception:
            # Fallback without no_grad if torch import fails (very unlikely)
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                do_sample=do_sample,
                use_cache=True,
            )

        # Decode
        output = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Best-effort: return only the assistant reply after the last template marker if present
        parts = re.split(r"\n?assistant:\s*", output, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return parts[-1].strip()
        return output.strip()

    def infer_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        cancel_event: Optional[threading.Event] = None,
    ):
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

        mm_messages, images, videos = self.build_mm_messages(messages)
        # Auto-compress if needed based on context budget
        mm_messages, ctx_info = self._auto_compress_if_needed(mm_messages, max_tokens)
        self.last_context_info = ctx_info

        text = self.processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        proc_kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos

        inputs = self.processor(**proc_kwargs)
        try:
            if str(getattr(self, "_resolved_device_map", "")).lower() == "cpu":
                inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}
            else:
                device = getattr(self.model, "device", None) or next(self.model.parameters()).device
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass

        do_sample = temperature is not None and float(temperature) > 0.0

        streamer = TextIteratorStreamer(
            getattr(self.processor, "tokenizer", None),
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=do_sample,
            use_cache=True,
            streamer=streamer,
        )

        # Optional cooperative cancellation via StoppingCriteria
        if cancel_event is not None:
            class _CancelCrit(StoppingCriteria):
                def __init__(self, ev: threading.Event):
                    self.ev = ev

                def __call__(self, input_ids, scores, **kwargs):
                    return bool(self.ev.is_set())

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_CancelCrit(cancel_event)])

        # Wrap generation with torch.no_grad() to avoid autograd overhead on CPU and reduce failure surface
        def _runner():
            try:
                import torch
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except Exception:
                # Let streamer finish gracefully even if generation throws
                pass

        th = threading.Thread(target=_runner)
        th.start()

        for piece in streamer:
            if piece:
                yield piece


# Simple in-memory resumable SSE session store + optional SQLite persistence
class _SSESession:
    def __init__(self, maxlen: int = 2048, ttl_seconds: int = 600):
        self.buffer: Deque[Tuple[int, str]] = deque(maxlen=maxlen)  # (idx, sse_line_block)
        self.last_idx: int = -1
        self.created: float = time.time()
        self.finished: bool = False
        self.cond = threading.Condition()
        self.thread: Optional[threading.Thread] = None
        self.ttl_seconds = ttl_seconds
        # Cancellation + client tracking
        self.cancel_event = threading.Event()
        self.listeners: int = 0
        self.cancel_timer = None  # type: ignore


class _SessionStore:
    def __init__(self, ttl_seconds: int = 600, max_sessions: int = 256):
        self._sessions: Dict[str, _SSESession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions

    def get_or_create(self, sid: str) -> _SSESession:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                sess = _SSESession(ttl_seconds=self._ttl)
                self._sessions[sid] = sess
            return sess

    def get(self, sid: str) -> Optional[_SSESession]:
        with self._lock:
            return self._sessions.get(sid)

    def gc(self):
        now = time.time()
        with self._lock:
            # remove expired
            expired = [k for k, v in self._sessions.items() if (now - v.created) > self._ttl or (v.finished and (now - v.created) > self._ttl / 4)]
            for k in expired:
                self._sessions.pop(k, None)
            # bound session count
            if len(self._sessions) > self._max_sessions:
                for k, _ in sorted(self._sessions.items(), key=lambda kv: kv[1].created)[: max(0, len(self._sessions) - self._max_sessions)]:
                    self._sessions.pop(k, None)


class _SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, created REAL, finished INTEGER DEFAULT 0)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS events (session_id TEXT, idx INTEGER, data TEXT, created REAL, PRIMARY KEY(session_id, idx))"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, idx)")
        self._conn.commit()

    def ensure_session(self, session_id: str, created: int):
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created, finished) VALUES (?, ?, 0)",
                (session_id, float(created)),
            )
            self._conn.commit()

    def append_event(self, session_id: str, idx: int, payload: Dict[str, Any]):
        data = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events(session_id, idx, data, created) VALUES (?, ?, ?, ?)",
                (session_id, idx, data, time.time()),
            )
            self._conn.commit()

    def get_events_after(self, session_id: str, last_idx: int) -> List[Tuple[int, str]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT idx, data FROM events WHERE session_id=? AND idx>? ORDER BY idx ASC", (session_id, last_idx)
            )
            return [(int(r[0]), str(r[1])) for r in cur.fetchall()]

    def mark_finished(self, session_id: str):
        with self._lock:
            self._conn.execute("UPDATE sessions SET finished=1 WHERE session_id=?", (session_id,))
            self._conn.commit()

    def session_meta(self, session_id: str) -> Tuple[bool, int]:
        with self._lock:
            row = self._conn.execute("SELECT finished FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            finished = bool(row[0]) if row else False
            row2 = self._conn.execute("SELECT MAX(idx) FROM events WHERE session_id=?", (session_id,)).fetchone()
            last_idx = int(row2[0]) if row2 and row2[0] is not None else -1
            return finished, last_idx

    def gc(self, ttl_seconds: int):
        cutoff = time.time() - float(ttl_seconds)
        with self._lock:
            cur = self._conn.execute("SELECT session_id FROM sessions WHERE finished=1 AND created<?", (cutoff,))
            ids = [r[0] for r in cur.fetchall()]
            for sid in ids:
                self._conn.execute("DELETE FROM events WHERE session_id=?", (sid,))
                self._conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
            self._conn.commit()


def _sse_event(session_id: str, idx: int, payload: Dict[str, Any]) -> str:
    # Include SSE id line so clients can send Last-Event-ID to resume.
    return f"id: {session_id}:{idx}\n" + f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_STORE = _SessionStore()
_DB_STORE = _SQLiteStore(SESSIONS_DB_PATH) if PERSIST_SESSIONS else None

# FastAPI app and OpenAPI tags
tags_metadata = [
    {"name": "meta", "description": "Service metadata and OpenAPI schema"},
    {"name": "health", "description": "Readiness and runtime info including context window report"},
    {"name": "chat", "description": "OpenAI-compatible chat completions (non-stream and streaming SSE)"},
    {"name": "ocr", "description": "Optical Character Recognition endpoints"},
]

app = FastAPI(
    title="Qwen3-VL Inference Server",
    version="1.0.0",
    description="OpenAI-compatible inference server for Qwen3-VL with multimodal support, streaming SSE with resume, context auto-compression, and optional SQLite persistence.",
    openapi_tags=tags_metadata,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup hook is defined after get_engine() so globals are initialized first.
# Serve static web UI if present
_WEB_DIR = os.path.join(ROOT_DIR, "web")
if os.path.isdir(_WEB_DIR):
    try:
        app.mount("/web", StaticFiles(directory=_WEB_DIR, html=True), name="web")
    except Exception:
        pass

# Engine singletons
_engine: Optional[Engine] = None
_engine_error: Optional[str] = None


def get_engine() -> Engine:
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    try:
        model_id = DEFAULT_MODEL_ID
        _log(f"Preparing model '{model_id}' (HF_HOME={os.getenv('HF_HOME')}, cache={os.getenv('TRANSFORMERS_CACHE')})")
        local_repo_dir = prefetch_model_assets(model_id, HF_TOKEN)
        load_id = local_repo_dir if (local_repo_dir and os.path.exists(os.path.join(local_repo_dir, 'config.json'))) else model_id
        _log(f"Loading processor and model from: {load_id}")
        _engine = Engine(model_id=load_id, hf_token=HF_TOKEN)
        _engine_error = None
        _log(f"Model ready: {_engine.model_id}")
        return _engine
    except Exception as e:
        _engine_error = f"{type(e).__name__}: {e}"
        _log(f"Engine init failed: {_engine_error}")
        raise

# Eager-load model at startup after definitions so it downloads/checks before serving traffic.
@app.on_event("startup")
def _startup_load_model():
    # Initialize marketplace database
    try:
        from database import init_db
        init_db()
        print("[startup] Marketplace database initialized")
    except Exception as e:
        print(f"[startup] Database initialization failed: {e}")

    if EAGER_LOAD_MODEL:
        print("[startup] EAGER_LOAD_MODEL=1: initializing model...")
        print("[startup] OCR engine disabled - using image captioning instead")
        try:
            # Then initialize the model
            _ = get_engine()
            print("[startup] Model loaded:", _engine.model_id if _engine else "unknown")
        except Exception as e:
            # Log error but don't fail - allow server to start without model
            print("[startup] Initialization failed:", e)
            print("[startup] Server will start without full initialization")
    else:
        print("[startup] EAGER_LOAD_MODEL=0: skipping initialization")


@app.get("/", tags=["meta"], include_in_schema=False)
def root():
    """
    Serve the client web UI. The UI calls an external Hugging Face Space API
    (default is KillerKing93/Transformers-TextEngine-InferenceServer-OpenAPI-Compatible-V3) and does NOT
    use internal server endpoints for chat. You can change the base via the input
    field or ?api= query string in the page.
    """
    index_path = os.path.join(ROOT_DIR, "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html; charset=utf-8")
    # Inline minimal fallback to make root return an HTML page even if COPY failed
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Qwen3‑VL Chat</title></head>
    <body style="font-family:system-ui,Segoe UI,Roboto;padding:24px;background:#0f172a;color:#e2e8f0">
    <h2>Qwen3‑VL Chat UI</h2>
    <p>The static UI was not found inside the container. This page is a fallback.</p>
    <p>Try pulling the latest image or rebuilding the Space so that <code>/app/web/index.html</code> is present.</p>
    <p>Once copied, this URL will serve the full UI. For now you can open the raw UI file from the repo or call the API directly.</p>
    <ul>
      <li><a href="./docs" style="color:#93c5fd">Swagger UI</a></li>
      <li><a href="./openapi.yaml" style="color:#93c5fd">OpenAPI YAML</a></li>
    </ul>
    </body></html>"""
    return Response(html, media_type="text/html; charset=utf-8")


@app.get("/openapi.yaml", tags=["meta"])
def openapi_yaml():
    """Serve OpenAPI schema as YAML for tooling compatibility."""
    schema = app.openapi()
    yml = yaml.safe_dump(schema, sort_keys=False)
    return Response(yml, media_type="application/yaml")


@app.get("/health", tags=["health"], response_model=HealthResponse)
def health():
    ready = False
    err = None
    model_id = DEFAULT_MODEL_ID
    global _engine, _engine_error
    if _engine is not None:
        ready = True
        model_id = _engine.model_id
    elif _engine_error:
        err = _engine_error
    ctx = None
    try:
        if _engine is not None:
            ctx = _engine.get_context_report()
    except Exception:
        ctx = None
    return JSONResponse({"ok": True, "modelReady": ready, "modelId": model_id, "error": err, "context": ctx})


@app.post(
    "/v1/chat/completions",
    tags=["chat"],
    response_model=ChatCompletionResponse,
    responses={
        200: {
            "description": "When stream=true, the response is text/event-stream (SSE). When stream=false, JSON body matches ChatCompletionResponse.",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "examples": {
                        "sse": {
                            "summary": "SSE stream example",
                            "value": "id: sess-123:0\ndata: {\"id\":\"sess-123\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"
                        }
                    }
                }
            },
        }
    },
)
async def chat_completions(
    request: Request,
    body: ChatRequest,
    last_event_id: Optional[str] = Query(default=None, alias="last_event_id", description="Resume SSE from this id: 'session_id:index'"),
    last_event_id_header: Optional[str] = Header(default=None, alias="Last-Event-ID", convert_underscores=False, description="SSE resume id 'session_id:index'"),
):
    # Ensure engine is loaded
    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

    if not body or not isinstance(body.messages, list) or len(body.messages) == 0:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    max_tokens = int(body.max_tokens) if isinstance(body.max_tokens, int) else DEFAULT_MAX_TOKENS
    temperature = float(body.temperature) if body.temperature is not None else DEFAULT_TEMPERATURE
    do_stream = bool(body.stream)

    # Parse Last-Event-ID (header or ?last_event_id) and derive/align session_id
    le_id = last_event_id_header or last_event_id
    sid_from_header: Optional[str] = None
    last_idx_from_header: int = -1
    if le_id:
        try:
            sid_from_header, idx_str = le_id.split(":", 1)
            last_idx_from_header = int(idx_str)
        except Exception:
            sid_from_header = None
            last_idx_from_header = -1

    session_id = body.session_id or sid_from_header or f"sess-{uuid.uuid4().hex[:12]}"
    sess = _STORE.get_or_create(session_id)
    created_ts = int(sess.created)
    if _DB_STORE is not None:
        _DB_STORE.ensure_session(session_id, created_ts)

    if not do_stream:
        # Non-streaming path
        try:
            content = engine.infer(body.messages, max_tokens=max_tokens, temperature=temperature)
        except ValueError as e:
            # Parsing/user payload errors from engine -> HTTP 400
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        now = int(time.time())
        prompt_tokens = int((engine.last_context_info or {}).get("prompt_tokens") or 0)
        completion_tokens = max(1, len((content or "").split()))
        total_tokens = prompt_tokens + completion_tokens
        resp: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": now,
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "context": engine.last_context_info or {},
        }
        return JSONResponse(resp)

    # Streaming SSE with resumable support
    def sse_generator():
        # Manage listener count and cancel timer
        sess.listeners += 1
        try:
            # Cancel any pending cancel timer when a listener attaches
            if getattr(sess, "cancel_timer", None):
                try:
                    sess.cancel_timer.cancel()
                except Exception:
                    pass
                sess.cancel_timer = None

            # Replay only when a valid Last-Event-ID is provided for this same session
            do_replay = bool(sid_from_header) and (sid_from_header == session_id)
            if do_replay:
                replay_from = last_idx_from_header
                # First try in-memory buffer
                for idx, block in list(sess.buffer):
                    if idx > replay_from:
                        yield block.encode("utf-8")
                # Optionally pull from SQLite persistence
                if _DB_STORE is not None:
                    try:
                        for idx, data in _DB_STORE.get_events_after(session_id, replay_from):
                            block = f"id: {session_id}:{idx}\n" + f"data: {data}\n\n"
                            yield block.encode("utf-8")
                    except Exception:
                        pass
                if sess.finished:
                    # Already finished; send terminal and exit
                    yield b"data: [DONE]\n\n"
                    return

            # Fresh generation path
            # Helper to append to buffers and yield to client
            def push(payload: Dict[str, Any]):
                sess.last_idx += 1
                idx = sess.last_idx
                block = _sse_event(session_id, idx, payload)
                sess.buffer.append((idx, block))
                if _DB_STORE is not None:
                    try:
                        _DB_STORE.append_event(session_id, idx, payload)
                    except Exception:
                        pass
                return block

            # Initial assistant role delta
            head = {
                "id": session_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "system_fingerprint": "fastapi",
            }
            yield push(head).encode("utf-8")

            # Stream model pieces
            try:
                for piece in engine.infer_stream(
                    body.messages, max_tokens=max_tokens, temperature=temperature, cancel_event=sess.cancel_event
                ):
                    if not piece:
                        continue
                    payload = {
                        "id": session_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": engine.model_id,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield push(payload).encode("utf-8")
                    # Cooperative early-exit if cancel requested
                    if sess.cancel_event.is_set():
                        break
            except Exception:
                # On engine error, terminate gracefully
                pass

            # Finish chunk
            finish = {
                "id": session_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield push(finish).encode("utf-8")

        finally:
            # Mark finished and persist
            sess.finished = True
            if _DB_STORE is not None:
                try:
                    _DB_STORE.mark_finished(session_id)
                    # Optionally GC older finished sessions
                    _DB_STORE.gc(SESSIONS_TTL_SECONDS)
                except Exception:
                    pass

            # Always send terminal [DONE]
            yield b"data: [DONE]\n\n"

            # Listener bookkeeping and optional auto-cancel if all disconnect
            try:
                sess.listeners = max(0, sess.listeners - 1)
                if sess.listeners == 0 and CANCEL_AFTER_DISCONNECT_SECONDS > 0 and not sess.cancel_event.is_set():
                    def _later_cancel():
                        # If still no listeners, cancel
                        if sess.listeners == 0 and not sess.cancel_event.is_set():
                            sess.cancel_event.set()
                    sess.cancel_timer = threading.Timer(CANCEL_AFTER_DISCONNECT_SECONDS, _later_cancel)
                    sess.cancel_timer.daemon = True
                    sess.cancel_timer.start()
            except Exception:
                pass

            # In-memory store GC
            try:
                _STORE.gc()
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=headers)


# OCR endpoint disabled - use image captioning instead
# @app.post("/ktp-ocr/", tags=["ocr"])
# async def ktp_ocr(image: UploadFile = File(...)):
#     # OCR functionality disabled


# Image Captioning for Multimodal Chat
async def _generate_image_caption(image_bytes: bytes, context: Optional[str] = None) -> str:
    """Generate image description using BLIP-2 model"""
    print(f"[CAPTION] Generating image caption{' with context' if context else ''}")

    try:
        from image_caption import get_image_captioner
        captioner = get_image_captioner()

        # Generate caption
        if context:
            caption = captioner.caption_with_context(image_bytes, context)
        else:
            caption = captioner.caption_image(image_bytes)

        print(f"[CAPTION] Generated: {caption}")
        return caption

    except Exception as e:
        print(f"[CAPTION] Error: {e}")
        # Fallback to generic description
        return "An image was uploaded for context"
        print(f"[OCR] Engine ready: {engine.model_id}")
    except Exception as e:
        print(f"[OCR] Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

    # Enhanced prompt for better OCR extraction
    prompt = """Anda adalah sistem OCR untuk KTP Indonesia. Ekstrak semua data yang terlihat dari gambar KTP ini dengan sangat teliti.

PERATURAN PENTING:
1. Jawaban HANYA berupa JSON yang valid, tanpa teks tambahan
2. Jika data tidak terlihat jelas, gunakan string kosong ""
3. Pastikan format JSON sesuai persis dengan contoh
4. Baca teks dengan teliti dari gambar

FORMAT JSON YANG HARUS DIPERIKSA:

{
  "nik": "nomor NIK 16 digit lengkap",
  "nama": "nama lengkap sesuai KTP",
  "tempat_lahir": "kota/kabupaten tempat lahir",
  "tgl_lahir": "tanggal lahir format DD-MM-YYYY",
  "jenis_kelamin": "LAKI-LAKI atau PEREMPUAN",
  "alamat": {
    "name": "alamat lengkap tanpa RT/RW",
    "rt_rw": "RT/RW format 001/002",
    "kel_desa": "nama kelurahan/desa",
    "kecamatan": "nama kecamatan"
  },
  "agama": "agama sesuai KTP",
  "status_perkawinan": "status pernikahan",
  "pekerjaan": "pekerjaan sesuai KTP",
  "kewarganegaraan": "WNI atau WNA",
  "berlaku_hingga": "tanggal berlaku hingga atau SEUMUR HIDUP"
}

MULAI EKSTRAKSI DATA:"""

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": pil_image}
            ],
        }
    ]

    # Run inference in thread pool to avoid blocking
    print(f"[OCR] Submitting to thread pool (timeout: {OCR_TIMEOUT_SECONDS}s)...")
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(
        executor,
        functools.partial(engine.infer, messages, 2048, 0.0)  # Lower temperature for more consistent output
    )
    print(f"[OCR] Raw inference result (length: {len(content)} chars): {repr(content[:500])}...")

    # Extract JSON from the response
    json_str = None

    # First, try to find JSON in code blocks
    json_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        print(f"[OCR] Found JSON in code block: {repr(json_str[:200])}...")
    else:
        # Try to find JSON object directly
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            print(f"[OCR] Found JSON object: {repr(json_str[:200])}...")

    # If no JSON found, try to extract from assistant response
    if not json_str:
        # Split on assistant marker
        parts = re.split(r"\n?assistant:\s*", content, flags=re.IGNORECASE)
        if len(parts) >= 2:
            assistant_content = parts[-1].strip()
            # Remove thinking tags
            assistant_content = re.sub(r"<think>.*?</think>", "", assistant_content, flags=re.DOTALL).strip()
            assistant_content = re.sub(r"<thinking>.*?</thinking>", "", assistant_content, flags=re.DOTALL).strip()

            # Look for JSON in assistant content
            json_match = re.search(r"\{.*\}", assistant_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
                print(f"[OCR] Found JSON in assistant response: {repr(json_str[:200])}...")

    # Parse the JSON
    if json_str:
        try:
            response_data = json.loads(json_str)
            print(f"[OCR] Successfully parsed JSON with keys: {list(response_data.keys())}")

            # Validate the structure has required fields
            required_fields = ["nik", "nama", "tempat_lahir", "tgl_lahir", "jenis_kelamin", "alamat", "agama", "status_perkawinan", "pekerjaan", "kewarganegaraan", "berlaku_hingga"]
            if all(field in response_data for field in required_fields):
                print(f"[OCR] JSON structure is valid")
                return JSONResponse(content=response_data)
            else:
                print(f"[OCR] JSON missing required fields, using fallback")
        except json.JSONDecodeError as e:
            print(f"[OCR] JSON parsing failed: {e}")
            print(f"[OCR] Failed JSON string: {repr(json_str[:500])}")

    # Fallback: return default empty structure
    print(f"[OCR] Using default empty KTP structure as fallback")
    response_data = {
        "nik": "",
        "nama": "",
        "tempat_lahir": "",
        "tgl_lahir": "",
        "jenis_kelamin": "",
        "alamat": {
            "name": "",
            "rt_rw": "",
            "kel_desa": "",
            "kecamatan": ""
        },
        "agama": "",
        "status_perkawinan": "",
        "pekerjaan": "",
        "kewarganegaraan": "",
        "berlaku_hingga": ""
    }
    return JSONResponse(content=response_data)


def _parse_ktp_from_text(text_lines):
    """Parse KTP data from OCR extracted text lines"""
    print(f"[PARSE] Starting to parse {len(text_lines)} text lines")
    ktp_data = {
        "nik": "",
        "nama": "",
        "tempat_lahir": "",
        "tgl_lahir": "",
        "jenis_kelamin": "",
        "alamat": {
            "name": "",
            "rt_rw": "",
            "kel_desa": "",
            "kecamatan": ""
        },
        "agama": "",
        "status_perkawinan": "",
        "pekerjaan": "",
        "kewarganegaraan": "",
        "berlaku_hingga": ""
    }

    print(f"[PARSE] Raw text lines: {text_lines}")

    # Extract NIK (16 digits) - handle both formats: "NIK 3506042602660001" and "NIK : 3506042602660001"
    nik_pattern = re.compile(r'\b(\d{16})\b')
    for i, line in enumerate(text_lines):
        nik_match = nik_pattern.search(line)
        if nik_match:
            ktp_data["nik"] = nik_match.group(1)
            print(f"[PARSE] Found NIK: {ktp_data['nik']}")
            break
        # Also check if NIK is on the next line after "NIK" label
        elif "NIK" in line.upper() and i + 1 < len(text_lines):
            next_line = text_lines[i + 1]
            nik_match = nik_pattern.search(next_line)
            if nik_match:
                ktp_data["nik"] = nik_match.group(1)
                print(f"[PARSE] Found NIK (next line): {ktp_data['nik']}")
                break

    # Extract Nama - look for "Nama" followed by name
    for i, line in enumerate(text_lines):
        if "NAMA" in line.upper():
            # Check if name is on the same line after ":"
            if ":" in line:
                name = line.split(":", 1)[1].strip()
                if name and len(name) > 1:
                    ktp_data["nama"] = name.title()
                    print(f"[PARSE] Found Nama (same line): {ktp_data['nama']}")
                    break
            # Check if name is on the next line (with or without colon prefix)
            elif i + 1 < len(text_lines):
                next_line = text_lines[i + 1]
                name = next_line.lstrip(':').strip()
                if name and len(name) > 1:
                    ktp_data["nama"] = name.title()
                    print(f"[PARSE] Found Nama (next line): {ktp_data['nama']}")
                    break

    # Extract Tempat/Tgl Lahir
    for i, line in enumerate(text_lines):
        if "LAHIR" in line.upper():
            # Look in current line and next few lines for date and place
            search_lines = [line]
            if i + 1 < len(text_lines):
                search_lines.append(text_lines[i + 1])
            if i + 2 < len(text_lines):
                search_lines.append(text_lines[i + 2])

            combined_text = " ".join(search_lines)
            print(f"[PARSE] Birth search text: '{combined_text}'")

            # Extract date pattern DD-MM-YYYY or DD/MM/YYYY (handle colon prefix)
            date_match = re.search(r':?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', combined_text)
            if date_match:
                ktp_data["tgl_lahir"] = date_match.group(1).replace('/', '-')
                print(f"[PARSE] Found Tgl Lahir: {ktp_data['tgl_lahir']}")

            # Extract place (usually before comma, handle colon prefix)
            place_match = re.search(r':?\s*([A-Z\s]+?)\s*,\s*\d', combined_text)
            if place_match:
                ktp_data["tempat_lahir"] = place_match.group(1).strip().title()
                print(f"[PARSE] Found Tempat Lahir: {ktp_data['tempat_lahir']}")
            break

    # Extract Jenis Kelamin
    for i, line in enumerate(text_lines):
        if "JENIS KELAMIN" in line.upper() or "KELAMIN" in line.upper():
            # Check current line for gender
            if "LAKI" in line.upper():
                ktp_data["jenis_kelamin"] = "LAKI-LAKI"
                print(f"[PARSE] Found Jenis Kelamin (same line): {ktp_data['jenis_kelamin']}")
                break
            elif "PEREMPUAN" in line.upper():
                ktp_data["jenis_kelamin"] = "PEREMPUAN"
                print(f"[PARSE] Found Jenis Kelamin (same line): {ktp_data['jenis_kelamin']}")
                break
            # Check next line for gender (with colon prefix)
            elif i + 1 < len(text_lines):
                next_line = text_lines[i + 1]
                gender_text = next_line.lstrip(':').strip().upper()
                print(f"[PARSE] Checking gender in next line: '{gender_text}'")
                if "LAKI" in gender_text:
                    ktp_data["jenis_kelamin"] = "LAKI-LAKI"
                    print(f"[PARSE] Found Jenis Kelamin (next line): {ktp_data['jenis_kelamin']}")
                    break
                elif "PEREMPUAN" in gender_text:
                    ktp_data["jenis_kelamin"] = "PEREMPUAN"
                    print(f"[PARSE] Found Jenis Kelamin (next line): {ktp_data['jenis_kelamin']}")
                    break

    # Extract Alamat
    alamat_start = False
    alamat_lines = []
    main_address_line = ""
    for i, line in enumerate(text_lines):
        if "ALAMAT" in line.upper():
            alamat_start = True
            # Check if address is on the same line
            alamat_match = re.search(r'ALAMAT\s*:\s*(.+)', line, re.IGNORECASE)
            if alamat_match:
                main_address_line = alamat_match.group(1).strip()
            # Check next line for address (with colon prefix)
            elif i + 1 < len(text_lines):
                next_line = text_lines[i + 1]
                if next_line.startswith(':'):
                    main_address_line = next_line[1:].strip()  # Remove colon prefix
            continue
        elif alamat_start and any(keyword in line.upper() for keyword in ["AGAMA", "STATUS", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU"]):
            break
        elif alamat_start:
            alamat_lines.append(line.strip())

    if alamat_lines or main_address_line:
        combined_alamat = (main_address_line + " " + " ".join(alamat_lines)).strip()
        print(f"[PARSE] Found Alamat combined: {combined_alamat}")

        # Extract RT/RW
        rt_rw_match = re.search(r'RT/RW\s*[:\-]?\s*(\d{1,3})/(\d{1,3})', combined_alamat, re.IGNORECASE)
        if rt_rw_match:
            ktp_data["alamat"]["rt_rw"] = f"{rt_rw_match.group(1).zfill(3)}/{rt_rw_match.group(2).zfill(3)}"
            print(f"[PARSE] Found RT/RW: {ktp_data['alamat']['rt_rw']}")

        # Extract Kel/Desa
        kel_desa_match = re.search(r'KEL/DESA\s*[:\-]?\s*(.+?)(?:\s*KECAMATAN|\s*Agama|\s*Status|\s*Pekerjaan|\s*$)', combined_alamat, re.IGNORECASE)
        if kel_desa_match:
            kel_desa = kel_desa_match.group(1).strip()
            ktp_data["alamat"]["kel_desa"] = kel_desa.title()
            print(f"[PARSE] Found Kel/Desa: {ktp_data['alamat']['kel_desa']}")

        # Extract Kecamatan
        kecamatan_match = re.search(r'KECAMATAN\s*[:\-]?\s*(.+)', combined_alamat, re.IGNORECASE)
        if kecamatan_match:
            kecamatan = kecamatan_match.group(1).strip()
            ktp_data["alamat"]["kecamatan"] = kecamatan.title()
            print(f"[PARSE] Found Kecamatan: {ktp_data['alamat']['kecamatan']}")

        # Set the main address
        ktp_data["alamat"]["name"] = main_address_line
        print(f"[PARSE] Main address: '{ktp_data['alamat']['name']}'")

    # Extract Agama
    for i, line in enumerate(text_lines):
        if "AGAMA" in line.upper():
            religions = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDHA", "KONGHUCU"]
            # Check current line for religion
            for religion in religions:
                if religion in line.upper():
                    ktp_data["agama"] = religion.title()
                    print(f"[PARSE] Found Agama (same line): {ktp_data['agama']}")
                    break
            else:
                # Check next line for religion (with colon prefix)
                if i + 1 < len(text_lines):
                    next_line = text_lines[i + 1]
                    religion_text = next_line.lstrip(':').strip().upper()
                    for religion in religions:
                        if religion in religion_text:
                            ktp_data["agama"] = religion.title()
                            print(f"[PARSE] Found Agama (next line): {ktp_data['agama']}")
                            break
            break

    # Extract Status Perkawinan
    for line in text_lines:
        if "STATUS" in line.upper() and "PERKAWINAN" in line.upper():
            # Handle combined format like "StatusPerkawinan:KAWIN"
            status_match = re.search(r'STATUSPERKAWINAN\s*:\s*(.+)', line, re.IGNORECASE)
            if status_match:
                status_text = status_match.group(1).strip().upper()
                if "KAWIN" in status_text or "MENIKAH" in status_text:
                    ktp_data["status_perkawinan"] = "Kawin"
                elif "BELUM" in status_text:
                    ktp_data["status_perkawinan"] = "Belum Kawin"
                elif "CERAI" in status_text:
                    ktp_data["status_perkawinan"] = "Cerai"
                print(f"[PARSE] Found Status Perkawinan (combined): {ktp_data['status_perkawinan']}")
                break
            # Handle separate format
            elif "KAWIN" in line.upper() or "MENIKAH" in line.upper():
                ktp_data["status_perkawinan"] = "Kawin"
            elif "BELUM" in line.upper():
                ktp_data["status_perkawinan"] = "Belum Kawin"
            elif "CERAI" in line.upper():
                ktp_data["status_perkawinan"] = "Cerai"
            print(f"[PARSE] Found Status Perkawinan: {ktp_data['status_perkawinan']}")
            break

    # Extract Pekerjaan
    for i, line in enumerate(text_lines):
        if "PEKERJAAN" in line.upper():
            job_match = re.search(r'PEKERJAAN\s*[:\-]?\s*([A-Z\s]+)', line, re.IGNORECASE)
            if job_match:
                ktp_data["pekerjaan"] = job_match.group(1).strip().title()
                print(f"[PARSE] Found Pekerjaan (same line): {ktp_data['pekerjaan']}")
                break
            # Check next line for job (with colon prefix)
            elif i + 1 < len(text_lines):
                next_line = text_lines[i + 1]
                job_text = next_line.lstrip(':').strip()
                if job_text:
                    ktp_data["pekerjaan"] = job_text.title()
                    print(f"[PARSE] Found Pekerjaan (next line): {ktp_data['pekerjaan']}")
                    break

    # Extract Kewarganegaraan
    for line in text_lines:
        if "KEWARGANEGARAAN" in line.upper():
            # Handle combined format like "Kewarganegaraan:WNI"
            kewarganegaraan_match = re.search(r'KEWARGANEGARAAN\s*:\s*(.+)', line, re.IGNORECASE)
            if kewarganegaraan_match:
                nationality = kewarganegaraan_match.group(1).strip().upper()
                if "WNI" in nationality:
                    ktp_data["kewarganegaraan"] = "Wni"
                elif "WNA" in nationality:
                    ktp_data["kewarganegaraan"] = "Wna"
                print(f"[PARSE] Found Kewarganegaraan (combined): {ktp_data['kewarganegaraan']}")
                break
            # Handle separate format
            elif "WNI" in line.upper():
                ktp_data["kewarganegaraan"] = "Wni"
            elif "WNA" in line.upper():
                ktp_data["kewarganegaraan"] = "Wna"
            print(f"[PARSE] Found Kewarganegaraan: {ktp_data['kewarganegaraan']}")
            break

    # Extract Berlaku Hingga
    for i, line in enumerate(text_lines):
        if "BERLAKU" in line.upper() and "HINGGA" in line.upper():
            if "SEUMUR" in line.upper() and "HIDUP" in line.upper():
                ktp_data["berlaku_hingga"] = "Seumur Hidup"
                print(f"[PARSE] Found Berlaku Hingga (same line): {ktp_data['berlaku_hingga']}")
                break
            else:
                # Try to extract date from current line
                date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', line)
                if date_match:
                    ktp_data["berlaku_hingga"] = date_match.group(1).replace('/', '-')
                    print(f"[PARSE] Found Berlaku Hingga date (same line): {ktp_data['berlaku_hingga']}")
                    break
                # Check next line for date (with colon prefix)
                elif i + 1 < len(text_lines):
                    next_line = text_lines[i + 1]
                    date_text = next_line.lstrip(':').strip()
                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', date_text)
                    if date_match:
                        ktp_data["berlaku_hingga"] = date_match.group(1).replace('/', '-')
                        print(f"[PARSE] Found Berlaku Hingga date (next line): {ktp_data['berlaku_hingga']}")
                        break

    print(f"[PARSE] Final parsed data: {ktp_data}")
    return ktp_data


@app.post("/v1/cancel/{session_id}", tags=["chat"], response_model=CancelResponse, summary="Cancel a streaming session")
def cancel_session(session_id: str):
    sess = _STORE.get(session_id)
    if sess is not None:
        try:
            sess.cancel_event.set()
            sess.finished = True
            if _DB_STORE is not None:
                _DB_STORE.mark_finished(session_id)
        except Exception:
            pass
    return JSONResponse({"ok": True, "session_id": session_id})


# ============================================================================
# AUTHENTICATION API ENDPOINTS
# ============================================================================

from auth import (
    hash_password,
    verify_password,
    create_tokens_for_user,
    get_current_user,
    get_current_active_user,
    require_admin,
    require_supplier,
    Token,
    UserLogin,
    refresh_access_token
)
from database import get_db
from sqlalchemy.orm import Session
from models import Supplier, Product, User, Conversation, Message
from utils import haversine_distance, sort_by_distance, extract_location_query, build_ai_context

class AuthUserRegister(BaseModel):
    name: str
    email: str
    password: str
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    role: str = "user"  # user, supplier, admin


@app.post("/api/auth/register", tags=["auth"], response_model=Token)
def register_auth_user(user: AuthUserRegister, db: Session = Depends(get_db)):
    """
    Register new user with authentication.

    Creates user account with hashed password and returns JWT tokens.
    Role can be: user, supplier, or admin.
    """
    # Check if email exists
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = hash_password(user.password)

    # Create user
    db_user = User(
        name=user.name,
        email=user.email,
        hashed_password=hashed_password,
        city=user.city,
        latitude=user.latitude,
        longitude=user.longitude,
        role=user.role,
        ai_access_enabled=(user.role in ["supplier", "admin"])  # Premium by default
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create tokens
    tokens = create_tokens_for_user(
        email=db_user.email,
        role=db_user.role,
        user_id=db_user.id
    )

    return tokens


@app.post("/api/auth/login", tags=["auth"], response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login user and return JWT tokens.

    Authenticates user with email and password, returns access and refresh tokens.
    """
    # Find user
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Verify password
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Create tokens
    tokens = create_tokens_for_user(
        email=user.email,
        role=user.role,
        user_id=user.id
    )

    return tokens


@app.post("/api/auth/refresh", tags=["auth"])
def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.

    Returns new access token without requiring re-authentication.
    """
    try:
        new_access_token = refresh_access_token(refresh_token)
        return {"access_token": new_access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e


@app.get("/api/auth/me", tags=["auth"])
def get_me(current_user: dict = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Get current authenticated user profile.

    Requires valid JWT token in Authorization header.
    """
    user = db.query(User).filter(User.email == current_user["email"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "city": user.city,
        "role": user.role,
        "ai_access_enabled": user.ai_access_enabled,
        "created_at": user.created_at
    }


# ============================================================================
# MARKETPLACE API ENDPOINTS
# ============================================================================

# Pydantic models for marketplace API
class SupplierCreate(BaseModel):
    name: str = Field(..., description="Supplier contact name")
    business_name: str = Field(..., description="Business/company name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = None
    address: Optional[str] = None
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    city: str = Field(..., description="City name")
    province: Optional[str] = None

class SupplierResponse(BaseModel):
    id: int
    name: str
    business_name: str
    email: str
    phone: Optional[str]
    address: Optional[str]
    latitude: float
    longitude: float
    city: str
    province: Optional[str]
    is_active: bool
    registration_date: datetime

    model_config = ConfigDict(from_attributes=True)

class ProductCreate(BaseModel):
    name: str = Field(..., description="Product name")
    description: Optional[str] = None
    price: float = Field(..., gt=0, description="Price in IDR")
    stock_quantity: int = Field(..., ge=0, description="Available stock")
    category: str = Field(..., description="Product category")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    sku: Optional[str] = None

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    stock_quantity: Optional[int] = Field(None, ge=0)
    category: Optional[str] = None
    tags: Optional[str] = None
    is_available: Optional[bool] = None

class ProductResponse(BaseModel):
    id: int
    supplier_id: int
    name: str
    description: Optional[str]
    price: float
    stock_quantity: int
    category: str
    tags: Optional[str]
    sku: Optional[str]
    is_available: bool
    created_at: datetime
    supplier_name: Optional[str] = None
    distance_km: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

class UserCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = None
    province: Optional[str] = None
    ai_access_enabled: bool = False

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    city: Optional[str]
    province: Optional[str]
    ai_access_enabled: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AISearchRequest(BaseModel):
    user_id: int
    query: str = Field(..., description="Natural language query (e.g., 'laptop gaming Jakarta 10 juta')")
    session_id: Optional[str] = None

class AISearchResponse(BaseModel):
    session_id: str
    response: str
    products_found: int
    conversation_id: int


# Supplier endpoints
@app.post("/api/suppliers/register", tags=["marketplace"], response_model=SupplierResponse)
def register_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    """
    Register a new supplier.

    Suppliers can list and manage their products on the marketplace.
    Location (lat/lng) is required for distance-based product recommendations.
    """
    # Check if email already exists
    existing = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_supplier = Supplier(**supplier.model_dump())
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    return db_supplier


@app.get("/api/suppliers", tags=["marketplace"], response_model=List[SupplierResponse])
def list_suppliers(
    city: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all suppliers with optional city filter.
    """
    query = db.query(Supplier).filter(Supplier.is_active == True)
    if city:
        query = query.filter(Supplier.city.ilike(f"%{city}%"))
    suppliers = query.offset(skip).limit(limit).all()
    return suppliers


@app.get("/api/suppliers/{supplier_id}", tags=["marketplace"], response_model=SupplierResponse)
def get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    """Get supplier details by ID."""
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier


# Product endpoints
@app.post("/api/suppliers/{supplier_id}/products", tags=["marketplace"], response_model=ProductResponse)
def create_product(
    supplier_id: int,
    product: ProductCreate,
    db: Session = Depends(get_db)
):
    """
    Add a new product for a supplier.

    Requires supplier_id. Products are searchable by name, category, and tags.
    """
    # Verify supplier exists
    supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")

    db_product = Product(supplier_id=supplier_id, **product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    # Add supplier name to response
    response = ProductResponse.model_validate(db_product)
    response.supplier_name = supplier.business_name
    return response


@app.put("/api/products/{product_id}", tags=["marketplace"], response_model=ProductResponse)
def update_product(
    product_id: int,
    product_update: ProductUpdate,
    db: Session = Depends(get_db)
):
    """Update product details (price, stock, availability, etc.)."""
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Update only provided fields
    update_data = product_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_product, key, value)

    db.commit()
    db.refresh(db_product)

    # Add supplier name
    response = ProductResponse.model_validate(db_product)
    response.supplier_name = db_product.supplier.business_name
    return response


@app.get("/api/products", tags=["marketplace"], response_model=List[ProductResponse])
def list_products(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    available_only: bool = True,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List products with optional filters.

    Filters:
    - category: Product category (e.g., "laptop", "smartphone")
    - min_price, max_price: Price range in IDR
    - available_only: Only show products with stock > 0
    """
    query = db.query(Product)

    if available_only:
        query = query.filter(Product.is_available == True, Product.stock_quantity > 0)

    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))

    if min_price is not None:
        query = query.filter(Product.price >= min_price)

    if max_price is not None:
        query = query.filter(Product.price <= max_price)

    products = query.offset(skip).limit(limit).all()

    # Add supplier names
    results = []
    for p in products:
        resp = ProductResponse.model_validate(p)
        resp.supplier_name = p.supplier.business_name
        results.append(resp)

    return results


@app.get("/api/products/search", tags=["marketplace"], response_model=List[ProductResponse])
def search_products(
    q: str = Query(..., description="Search query for product name or tags"),
    category: Optional[str] = None,
    city: Optional[str] = None,
    max_price: Optional[float] = None,
    user_lat: Optional[float] = Query(None, description="User latitude for distance sorting"),
    user_lon: Optional[float] = Query(None, description="User longitude for distance sorting"),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Search products by keyword with location-aware sorting.

    If user_lat/user_lon provided, results are sorted by distance from user.
    """
    query = db.query(Product).join(Supplier).filter(
        Product.is_available == True,
        Product.stock_quantity > 0
    )

    # Keyword search in name and tags
    query = query.filter(
        (Product.name.ilike(f"%{q}%")) | (Product.tags.ilike(f"%{q}%"))
    )

    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))

    if city:
        query = query.filter(Supplier.city.ilike(f"%{city}%"))

    if max_price is not None:
        query = query.filter(Product.price <= max_price)

    products = query.offset(skip).limit(limit).all()

    # Convert to dict for distance sorting
    results = []
    for p in products:
        product_dict = {
            "id": p.id,
            "supplier_id": p.supplier_id,
            "name": p.name,
            "description": p.description,
            "price": p.price,
            "stock_quantity": p.stock_quantity,
            "category": p.category,
            "tags": p.tags,
            "sku": p.sku,
            "is_available": p.is_available,
            "created_at": p.created_at,
            "supplier_name": p.supplier.business_name,
            "latitude": p.supplier.latitude,
            "longitude": p.supplier.longitude
        }
        results.append(product_dict)

    # Sort by distance if user location provided
    if user_lat is not None and user_lon is not None:
        results = sort_by_distance(results, user_lat, user_lon)

    # Convert back to Pydantic models
    return [ProductResponse(**r) for r in results]


# User endpoints
@app.post("/api/users/register", tags=["marketplace"], response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.

    Users with ai_access_enabled=True can use the AI assistant for product recommendations.
    """
    # Check if email already exists
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_user = User(**user.model_dump())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/api/users/{user_id}", tags=["marketplace"], response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user details by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# AI-powered search endpoint
@app.post("/api/chat/search", tags=["marketplace"], response_model=AISearchResponse)
def ai_powered_search(request: AISearchRequest, db: Session = Depends(get_db)):
    """
    AI-powered product search with natural language.

    Example queries:
    - "Saya butuh laptop gaming di Jakarta, budget 10 juta"
    - "smartphone murah Bandung"
    - "laptop untuk programming, yang bagus apa?"

    The AI will:
    1. Parse the query to extract intent, location, budget
    2. Search relevant products from database
    3. Sort by distance from user
    4. Provide personalized recommendations with reasoning
    """
    # Verify user exists and has AI access
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.ai_access_enabled:
        raise HTTPException(status_code=403, detail="AI access not enabled for this user")

    # Extract search parameters from query
    query_lower = request.query.lower()

    # Extract category keywords
    category = None
    for cat in ["laptop", "smartphone", "tablet", "monitor", "keyboard", "mouse", "printer"]:
        if cat in query_lower:
            category = cat
            break

    # Extract price/budget
    max_price = None
    price_patterns = [
        r'budget\s+(\d+)\s*(?:juta|jt)',
        r'(\d+)\s*(?:juta|jt)',
        r'[<≤]\s*(\d+)\s*(?:juta|jt)',
        r'max\s+(\d+)\s*(?:juta|jt)'
    ]
    for pattern in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            max_price = float(match.group(1)) * 1_000_000  # Convert juta to full number
            break

    # Extract city from query or use user's city
    city = extract_location_query(request.query)
    if not city and user.city:
        city = user.city

    # Search products in database
    query_db = db.query(Product).join(Supplier).filter(
        Product.is_available == True,
        Product.stock_quantity > 0
    )

    if category:
        query_db = query_db.filter(Product.category.ilike(f"%{category}%"))

    if max_price:
        query_db = query_db.filter(Product.price <= max_price)

    if city:
        query_db = query_db.filter(Supplier.city.ilike(f"%{city}%"))

    products = query_db.limit(10).all()

    # Convert to dict and sort by distance
    product_list = []
    for p in products:
        product_dict = {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "price": p.price,
            "stock_quantity": p.stock_quantity,
            "category": p.category,
            "tags": p.tags,
            "supplier_name": p.supplier.business_name,
            "latitude": p.supplier.latitude,
            "longitude": p.supplier.longitude
        }
        product_list.append(product_dict)

    # Sort by distance if user has location
    if user.latitude and user.longitude:
        product_list = sort_by_distance(product_list, user.latitude, user.longitude)

    # Build AI context
    ai_context = build_ai_context(product_list, request.query)

    # Get or create conversation
    session_id = request.session_id or f"user_{user.id}_{int(datetime.utcnow().timestamp())}"
    conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()

    if not conversation:
        conversation = Conversation(
            user_id=user.id,
            session_id=session_id,
            title=request.query[:50]  # First 50 chars as title
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.query
    )
    db.add(user_message)

    # Call AI model with context
    try:
        engine = get_engine()

        # Build messages with context injection
        messages = [
            {
                "role": "system",
                "content": ai_context
            },
            {
                "role": "user",
                "content": request.query
            }
        ]

        # Generate AI response
        ai_response = engine.infer(messages, max_tokens=512, temperature=0.7)

        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response
        )
        db.add(assistant_message)
        db.commit()

        return AISearchResponse(
            session_id=session_id,
            response=ai_response,
            products_found=len(product_list),
            conversation_id=conversation.id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI inference failed: {str(e)}")


# ============================================================================
# MULTIMODAL IMAGE CAPTIONING ENDPOINTS
# ============================================================================

class ImageCaptionRequest(BaseModel):
    context: Optional[str] = None
    max_length: int = 50
    num_beams: int = 5

class ImageCaptionResponse(BaseModel):
    caption: str
    model_info: dict

@app.post("/api/image-caption", tags=["multimodal"], response_model=ImageCaptionResponse)
async def caption_image_endpoint(
    image: UploadFile = File(...),
    context: Optional[str] = None,
    max_length: int = 50,
    num_beams: int = 5
):
    """
    Generate image description using BLIP-2 model

    Uploads an image and returns a natural language description.
    Can be used for:
    - Product catalog descriptions
    - Image search and discovery
    - Visual content understanding
    """
    print(f"[CAPTION] Processing image: {image.filename}, content_type: {image.content_type}")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image contents
        contents = await image.read()
        print(f"[CAPTION] Image size: {len(contents)} bytes")

        # Generate caption
        caption = await _generate_image_caption(contents, context)

        # Get model info
        from image_caption import get_image_captioner
        captioner = get_image_captioner()
        model_info = captioner.get_model_info()

        print(f"[CAPTION] Successfully generated caption: {caption[:100]}...")

        return ImageCaptionResponse(
            caption=caption,
            model_info=model_info
        )

    except Exception as e:
        print(f"[CAPTION] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image captioning failed: {str(e)}")

@app.post("/api/chat-with-image", tags=["multimodal"])
async def chat_with_image(
    image: UploadFile = File(...),
    message: str = Query(..., description="Chat message about the image"),
    user_id: Optional[int] = Query(None, description="User ID for conversation tracking")
):
    """
    Chat with AI about an uploaded image

    Upload an image and ask questions about it. The AI will:
    1. Generate a description of the image using BLIP-2
    2. Answer your questions about the image using Qwen3-4B

    Use cases:
    - "Describe this product"
    - "What do you see in this image?"
    - "Is this item suitable for [purpose]?"
    """
    print(f"[CHAT-IMG] Processing image chat: {image.filename}, user_id: {user_id}")
    print(f"[CHAT-IMG] User message: {message}")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image contents
        contents = await image.read()
        print(f"[CHAT-IMG] Image size: {len(contents)} bytes")

        # Generate image caption
        image_description = await _generate_image_caption(contents)
        print(f"[CHAT-IMG] Image description: {image_description}")

        # Get AI engine
        engine = get_engine()

        # Prepare prompt for AI
        prompt = f"""The user has uploaded an image and asked: {message}

Image Description: {image_description}

Please answer the user's question about the image based on the description provided. Be helpful and specific."""

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Generate AI response
        response = engine.infer(messages, 1024, 0.7)

        # Save conversation if user_id provided (marketplace feature)
        if user_id:
            from database import SessionLocal
            from models import Conversation, Message

            db = SessionLocal()
            try:
                # Get or create conversation
                session_id = f"chat_img_{user_id}"
                conversation = db.query(Conversation).filter(
                    Conversation.session_id == session_id
                ).first()

                if not conversation:
                    conversation = Conversation(session_id=session_id, user_id=user_id)
                    db.add(conversation)
                    db.commit()
                    db.refresh(conversation)

                # Save user message
                user_msg = Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=f"[Image uploaded] {message}"
                )
                db.add(user_msg)

                # Save AI response
                ai_msg = Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=response
                )
                db.add(ai_msg)
                db.commit()

            except Exception as e:
                print(f"[CHAT-IMG] Failed to save conversation: {e}")
            finally:
                db.close()

        print(f"[CHAT-IMG] AI response: {response[:200]}...")

        return {
            "user_message": message,
            "image_description": image_description,
            "ai_response": response,
            "conversation_saved": user_id is not None
        }

    except Exception as e:
        print(f"[CHAT-IMG] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image chat failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import socket
    import time

    # Check if port is already in use and kill existing process
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    def kill_existing_processes():
        try:
            import subprocess
            import sys
            if sys.platform == "win32":
                # Windows: use taskkill
                subprocess.run(["taskkill", "/F", "/IM", "python.exe"], check=False, capture_output=True)
                subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], check=False, capture_output=True)
            else:
                # Unix-like systems
                subprocess.run(["pkill", "-f", "uvicorn"], check=False, capture_output=True)
                subprocess.run(["pkill", "-f", "python.*main.py"], check=False, capture_output=True)
            return True
        except Exception as e:
            print(f"[startup] Failed to kill existing process: {e}")
            return False

    # Try to free the port
    max_attempts = 3
    for attempt in range(max_attempts):
        if is_port_in_use(PORT):
            print(f"[startup] Port {PORT} is already in use (attempt {attempt + 1}/{max_attempts}). Attempting to kill existing process...")
            if kill_existing_processes():
                print(f"[startup] Killed existing processes on port {PORT}")
                time.sleep(2)  # Wait for port to be freed
                if not is_port_in_use(PORT):
                    print(f"[startup] Port {PORT} is now free")
                    break
            else:
                print(f"[startup] Failed to kill processes, waiting...")
                time.sleep(3)
        else:
            print(f"[startup] Port {PORT} is free")
            break
    else:
        print(f"[startup] Could not free port {PORT} after {max_attempts} attempts")
        print("[startup] Please manually kill any processes using this port")
        exit(1)

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)