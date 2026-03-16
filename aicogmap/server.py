"""
AI Cog Map — Visualization Server

Standalone FastAPI server that reads per-layer activation data from shared
memory and serves a real-time cognitive heatmap visualization.

Usage:
    aicogmap                            # http://localhost:7890
    aicogmap --port 7890 --shm-path /dev/shm/aicogmap-activations
    python -m aicogmap.server
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aicogmap.hook import DEFAULT_SHM_PATH, read_activations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("aicogmap.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AI Cog Map", docs_url=None, redoc_url=None)

_shm_path = DEFAULT_SHM_PATH
_history: list[dict[str, Any]] = []
_history_max = 60
_prev_norms: list[float] | None = None


@app.get("/api/activations")
async def api_activations():
    """Current per-layer activation norms."""
    global _prev_norms

    data = read_activations(_shm_path)
    if not data:
        return JSONResponse({"status": "no_data", "norms": [], "num_layers": 0})

    norms = data["norms"]
    num_layers = data["num_layers"]

    max_norm = max(norms) if norms else 1.0
    normalized = [n / max_norm if max_norm > 0 else 0 for n in norms]

    deltas = []
    if _prev_norms and len(_prev_norms) == len(norms):
        for i in range(len(norms)):
            deltas.append(abs(norms[i] - _prev_norms[i]))
    _prev_norms = norms[:]

    max_delta = max(deltas) if deltas else 1.0
    norm_deltas = [d / max_delta if max_delta > 0 else 0 for d in deltas] if deltas else []

    snapshot = {
        "t": time.time(),
        "mean_norm": sum(normalized) / len(normalized) if normalized else 0,
        "max_norm": max(normalized) if normalized else 0,
        "active_layers": sum(1 for n in normalized if n > 0.1),
    }
    _history.append(snapshot)
    if len(_history) > _history_max:
        _history.pop(0)

    return JSONResponse({
        "status": "ok",
        "num_layers": num_layers,
        "age_s": data["age_s"],
        "norms": normalized,
        "raw_norms": norms,
        "deltas": norm_deltas,
        "history": _history,
    })


@app.get("/api/health")
async def api_health():
    data = read_activations(_shm_path)
    connected = data is not None and data["age_s"] < 10
    return {
        "status": "ok" if connected else "disconnected",
        "service": "aicogmap",
        "shm_path": _shm_path,
        "connected": connected,
        "num_layers": data["num_layers"] if data else 0,
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Cog Map — Cognitive Visualization Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7890)
    parser.add_argument("--shm-path", default=DEFAULT_SHM_PATH,
                        help="Path to shared memory activation file")
    args = parser.parse_args()

    global _shm_path
    _shm_path = args.shm_path

    log.info("AI Cog Map server starting on %s:%d (shm: %s)", args.host, args.port, _shm_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
