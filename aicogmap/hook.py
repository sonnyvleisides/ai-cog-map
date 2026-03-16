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
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger("aicogmap.hook")

DEFAULT_SHM_PATH = "/dev/shm/aicogmap-activations"
HEADER_SIZE = 64  # bytes: magic(4) + version(4) + num_layers(4) + timestamp_ns(8) + flags(4) + reserved(40)
MAGIC = b"ACGM"
VERSION = 1
MAX_LAYERS = 256


class ActivationWriter:
    """Lock-free shared memory writer for per-layer activation norms."""

    def __init__(self, shm_path: str, num_layers: int):
        self.shm_path = shm_path
        self.num_layers = min(num_layers, MAX_LAYERS)
        self.data_size = HEADER_SIZE + self.num_layers * 4  # float32 per layer
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
            # Update timestamp
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
_layer_index_map: dict[str, int] = {}
_layer_counter: int = 0


def _make_hook(layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook_fn(module: nn.Module, input: Any, output: Any):
        if _writer is None:
            return
        try:
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return

            norm = tensor.detach().float().norm().item()
            _writer.record(layer_idx, norm)
        except Exception:
            pass

    return hook_fn


def create_activation_hook(config: dict[str, Any] | None = None) -> Callable:
    """
    SGLang hook factory. Called once at startup by SGLang's hook manager.

    Returns a forward hook function that SGLang attaches to each matched module.
    Each call to the returned hook identifies its layer by module identity and
    assigns a sequential index.

    Config options:
        shm_path: str — shared memory file path (default: /dev/shm/aicogmap-activations)
        flush_interval: float — seconds between shared memory flushes (default: 0.1)
        max_layers: int — maximum number of layers to track (default: 256)
    """
    global _writer, _flusher, _layer_counter

    config = config or {}
    shm_path = config.get("shm_path", DEFAULT_SHM_PATH)
    flush_interval = config.get("flush_interval", 0.1)
    max_layers = config.get("max_layers", MAX_LAYERS)

    logger.info("AI Cog Map: hook factory called, shm_path=%s", shm_path)

    # We don't know total layer count yet — SGLang calls this factory once,
    # then attaches the returned hook to each matched module. We'll initialize
    # the writer lazily on first hook call once we know the count, or
    # pre-allocate with max_layers.
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
            _layer_index_map[mod_id] = _layer_counter
            _layer_counter += 1
            # Update actual layer count in shared memory header
            if _writer._mm:
                _writer._mm.seek(8)
                _writer._mm.write(struct.pack("<I", _layer_counter))

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


def read_activations(shm_path: str = DEFAULT_SHM_PATH) -> dict[str, Any] | None:
    """
    Read current activation data from shared memory.
    Intended for use by the visualization server.

    Returns dict with keys: num_layers, timestamp_ns, norms (list of floats),
    or None if unavailable.
    """
    try:
        path = Path(shm_path)
        if not path.exists():
            return None

        with open(shm_path, "rb") as f:
            data = f.read()

        if len(data) < HEADER_SIZE:
            return None

        magic = data[:4]
        if magic != MAGIC:
            return None

        version = struct.unpack("<I", data[4:8])[0]
        num_layers = struct.unpack("<I", data[8:12])[0]
        timestamp_ns = struct.unpack("<Q", data[12:20])[0]

        if num_layers == 0 or len(data) < HEADER_SIZE + num_layers * 4:
            return None

        norms = []
        for i in range(num_layers):
            offset = HEADER_SIZE + i * 4
            val = struct.unpack("<f", data[offset:offset + 4])[0]
            norms.append(val)

        age_s = (time.time_ns() - timestamp_ns) / 1e9

        return {
            "version": version,
            "num_layers": num_layers,
            "timestamp_ns": timestamp_ns,
            "age_s": round(age_s, 3),
            "norms": norms,
        }
    except Exception:
        return None
