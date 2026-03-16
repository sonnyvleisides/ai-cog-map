"""
AI Cog Map — Shared Memory Reader

Reads activation data written by the SGLang hook. No torch dependency.
"""

import struct
import time
from pathlib import Path
from typing import Any

DEFAULT_SHM_PATH = "/dev/shm/aicogmap-activations"
HEADER_SIZE = 64
MAGIC = b"ACGM"


def read_activations(shm_path: str = DEFAULT_SHM_PATH) -> dict[str, Any] | None:
    """
    Read current activation data from shared memory.

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

        if num_layers == 0:
            return None

        available_layers = min(num_layers, (len(data) - HEADER_SIZE) // 4)
        if available_layers == 0:
            return None

        norms = []
        for i in range(available_layers):
            offset = HEADER_SIZE + i * 4
            val = struct.unpack("<f", data[offset:offset + 4])[0]
            norms.append(val)

        age_s = (time.time_ns() - timestamp_ns) / 1e9

        return {
            "version": version,
            "num_layers": available_layers,
            "total_hooks": num_layers,
            "timestamp_ns": timestamp_ns,
            "age_s": round(age_s, 3),
            "norms": norms,
        }
    except Exception:
        return None
