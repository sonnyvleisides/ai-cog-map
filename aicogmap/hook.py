"""
AI Cog Map — SGLang Forward Hook Plugin

Captures per-layer activation norms during inference and writes them to
shared memory for real-time visualization. Designed as a non-invasive
SGLang forward hook — zero overhead when not installed, minimal overhead
when active (~0.1-0.5% of inference time).

Usage with SGLang:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-27B \
        --forward-hooks '[{
            "name": "aicogmap",
            "target_modules": ["model.layers.*"],
            "hook_factory": "aicogmap.hook:create_activation_hook",
            "config": {"shm_path": "/dev/shm/aicogmap-activations"}
        }]'
"""

import json
import logging
import mmap
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("aicogmap.hook")

DEFAULT_SHM_PATH = "/dev/shm/aicogmap-activations"
DEFAULT_META_PATH = "/dev/shm/aicogmap-metadata.json"
HEADER_SIZE = 64  # bytes: magic(4) + version(4) + num_layers(4) + timestamp_ns(8) + flags(4) + reserved(40)
MAGIC = b"ACGM"
VERSION = 2
MAX_LAYERS = 1024

# Module name → (component_type, component_category) mapping.
# Ordered from most specific to least — first match wins.
_COMPONENT_RULES: list[tuple[str, str, str]] = [
    (r"\.input_layernorm$",          "layernorm_input",  "norm"),
    (r"\.post_attention_layernorm$",  "layernorm_post",   "norm"),
    (r"\.linear_attn\.conv1d$",       "attention_conv",   "attention"),
    (r"\.linear_attn\.in_proj_qkv$",  "attention_qkv",    "attention"),
    (r"\.linear_attn\.in_proj_z$",    "attention_gate",   "attention"),
    (r"\.linear_attn\.in_proj_a$",    "attention_proj_a", "projection"),
    (r"\.linear_attn\.in_proj_b$",    "attention_proj_b", "projection"),
    (r"\.self_attn\.q_proj$",         "attention_qkv",    "attention"),
    (r"\.self_attn\.k_proj$",         "attention_qkv",    "attention"),
    (r"\.self_attn\.v_proj$",         "attention_qkv",    "attention"),
    (r"\.self_attn\.o_proj$",         "attention_proj_a", "projection"),
    (r"\.self_attn$",                 "attention",        "attention"),
    (r"\.linear_attn$",              "attention",        "attention"),
    (r"\.mlp\.gate_up_proj$",         "mlp_gate_up",      "mlp"),
    (r"\.mlp\.gate_proj$",            "mlp_gate_up",      "mlp"),
    (r"\.mlp\.up_proj$",              "mlp_gate_up",      "mlp"),
    (r"\.mlp\.down_proj$",            "mlp_down",         "mlp"),
    (r"\.mlp$",                       "mlp",              "mlp"),
]
_COMPILED_RULES = [(re.compile(pat), ctype, ccat) for pat, ctype, ccat in _COMPONENT_RULES]


def _classify_module(name: str) -> tuple[int | None, str, str]:
    """Extract (layer_index, component_type, component_category) from a module name."""
    layer_match = re.search(r"model\.layers\.(\d+)", name)
    layer_idx = int(layer_match.group(1)) if layer_match else None

    for pattern, ctype, ccat in _COMPILED_RULES:
        if pattern.search(name):
            return layer_idx, ctype, ccat

    if layer_match and name.rstrip().endswith(f"model.layers.{layer_idx}"):
        return layer_idx, "layer_output", "layer"

    return layer_idx, "unknown", "other"


class ActivationWriter:
    """Lock-free shared memory writer for per-layer activation norms."""

    def __init__(self, shm_path: str, num_layers: int):
        self.shm_path = shm_path
        self.num_layers = min(num_layers, MAX_LAYERS)
        self.data_size = HEADER_SIZE + self.num_layers * 4
        self._lock = threading.Lock()
        self._layer_norms = [0.0] * self.num_layers
        self._dirty = False
        self._fd = None
        self._mm = None
        self._init_shm()

    def _init_shm(self):
        try:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, self.data_size)
            self._fd = fd
            self._mm = mmap.mmap(fd, self.data_size)
            self._write_header()
            logger.info(
                "AI Cog Map: shared memory initialized at %s (%d layers, %d bytes)",
                self.shm_path, self.num_layers, self.data_size,
            )
        except Exception:
            logger.exception("AI Cog Map: failed to initialize shared memory")

    def _write_header(self):
        if not self._mm:
            return
        self._mm.seek(0)
        self._mm.write(MAGIC)
        self._mm.write(struct.pack("<I", VERSION))
        self._mm.write(struct.pack("<I", self.num_layers))
        self._mm.write(struct.pack("<Q", 0))
        self._mm.write(struct.pack("<I", 0))
        self._mm.write(b"\x00" * 40)

    def record(self, layer_idx: int, norm_value: float):
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        with self._lock:
            self._layer_norms[layer_idx] = norm_value
            self._dirty = True

    def flush(self):
        if not self._mm or not self._dirty:
            return
        with self._lock:
            self._mm.seek(HEADER_SIZE)
            for val in self._layer_norms:
                self._mm.write(struct.pack("<f", val))
            self._mm.seek(12)
            self._mm.write(struct.pack("<Q", time.time_ns()))
            self._dirty = False

    def close(self):
        if self._mm:
            self._mm.close()
        if self._fd is not None:
            os.close(self._fd)


class ActivationFlusher(threading.Thread):
    """Background thread that flushes accumulated norms to shared memory."""

    def __init__(self, writer: ActivationWriter, interval: float = 0.1):
        super().__init__(daemon=True, name="aicogmap-flusher")
        self.writer = writer
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.writer.flush()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


_writer: ActivationWriter | None = None
_flusher: ActivationFlusher | None = None
_layer_index_map: dict[int, int] = {}
_layer_counter: int = 0
_module_metadata: dict[int, dict[str, Any]] = {}
_meta_path: str = DEFAULT_META_PATH
_model_name_map: dict[int, str] | None = None


_registration_order_names: list[str] = []


def _build_expected_names(num_layers: int = 64) -> list[str]:
    """Build the expected module name list matching SGLang's registration order.

    SGLang iterates model.named_modules() which yields modules in definition
    order. This must match the model architecture exactly. For Qwen3.5 with
    hybrid GDN/Mamba attention, each layer has these sub-modules in order.
    """
    names = []
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        names.extend([
            prefix,
            f"{prefix}.linear_attn",
            f"{prefix}.linear_attn.conv1d",
            f"{prefix}.linear_attn.in_proj_qkv",
            f"{prefix}.linear_attn.in_proj_z",
            f"{prefix}.linear_attn.in_proj_b",
            f"{prefix}.linear_attn.in_proj_a",
            f"{prefix}.linear_attn.attn",
            f"{prefix}.linear_attn.norm",
            f"{prefix}.linear_attn.out_proj",
            f"{prefix}.mlp",
            f"{prefix}.mlp.gate_up_proj",
            f"{prefix}.mlp.down_proj",
            f"{prefix}.mlp.act_fn",
            f"{prefix}.input_layernorm",
            f"{prefix}.post_attention_layernorm",
        ])
    return names


def _build_name_map(module: nn.Module) -> dict[int, str]:
    """Not used — name resolution uses registration order instead."""
    return {}


def _resolve_module_name(registration_index: int) -> str:
    """Resolve name by registration order index."""
    if registration_index < len(_registration_order_names):
        return _registration_order_names[registration_index]
    return f"unknown_{registration_index}"


def _write_metadata_sidecar():
    """Write the metadata sidecar JSON file atomically."""
    meta_entries = []
    for mod_id, idx in sorted(_layer_index_map.items(), key=lambda x: x[1]):
        info = _module_metadata.get(mod_id, {})
        meta_entries.append({
            "index": idx,
            "layer": info.get("layer"),
            "type": info.get("type", "unknown"),
            "category": info.get("category", "other"),
            "full_name": info.get("full_name", ""),
        })

    sidecar = {
        "version": VERSION,
        "num_hooks": len(meta_entries),
        "written_at": time.time(),
        "hooks": meta_entries,
    }

    tmp_path = _meta_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(sidecar, f, separators=(",", ":"))
        os.replace(tmp_path, _meta_path)
    except Exception:
        logger.debug("AI Cog Map: failed to write metadata sidecar", exc_info=True)


def create_activation_hook(config: dict[str, Any] | None = None) -> Callable:
    """
    SGLang hook factory. Called once at startup by SGLang's hook manager.

    Returns a forward hook function that SGLang attaches to each matched module.
    Each call to the returned hook identifies its layer by module identity,
    assigns a sequential index, and classifies the module's component type.

    Config options:
        shm_path: str — shared memory file path (default: /dev/shm/aicogmap-activations)
        meta_path: str — metadata sidecar JSON path (default: /dev/shm/aicogmap-metadata.json)
        flush_interval: float — seconds between shared memory flushes (default: 0.1)
        max_layers: int — maximum number of layers to track (default: 1024)
    """
    global _writer, _flusher, _layer_counter, _meta_path, _registration_order_names

    config = config or {}
    shm_path = config.get("shm_path", DEFAULT_SHM_PATH)
    _meta_path = config.get("meta_path", DEFAULT_META_PATH)
    flush_interval = config.get("flush_interval", 0.1)
    max_layers = config.get("max_layers", MAX_LAYERS)
    num_transformer_layers = config.get("num_layers", 64)

    logger.info("AI Cog Map: hook factory called, shm_path=%s, meta_path=%s", shm_path, _meta_path)

    _registration_order_names = _build_expected_names(num_transformer_layers)
    logger.info("AI Cog Map: expected %d hook registrations for %d transformer layers",
                len(_registration_order_names), num_transformer_layers)

    _writer = ActivationWriter(shm_path, max_layers)
    _flusher = ActivationFlusher(_writer, flush_interval)
    _flusher.start()

    _layer_counter = 0

    def hook_fn(module: nn.Module, input: Any, output: Any):
        global _layer_counter
        if _writer is None:
            return

        mod_id = id(module)
        if mod_id not in _layer_index_map:
            idx = _layer_counter
            _layer_index_map[mod_id] = idx
            _layer_counter += 1

            if _writer._mm:
                _writer._mm.seek(8)
                _writer._mm.write(struct.pack("<I", _layer_counter))

            full_name = _resolve_module_name(idx)
            layer_idx, comp_type, comp_cat = _classify_module(full_name)
            _module_metadata[mod_id] = {
                "layer": layer_idx,
                "type": comp_type,
                "category": comp_cat,
                "full_name": full_name,
            }
            _write_metadata_sidecar()
            logger.debug(
                "AI Cog Map: hook %d → %s (layer=%s, type=%s, cat=%s)",
                idx, full_name, layer_idx, comp_type, comp_cat,
            )
        else:
            idx = _layer_index_map[mod_id]

        try:
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return

            norm = tensor.detach().float().norm().item()
            _writer.record(idx, norm)
        except Exception:
            pass

    return hook_fn
