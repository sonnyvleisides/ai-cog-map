# AI Cog Map

**AI Model Cognitive Mapping Tool** — Real-time transformer layer activation visualization for SGLang inference servers.

Watch your AI model think. AI Cog Map captures per-layer activation patterns during inference and renders them as a live heatmap, showing which layers of the transformer are doing meaningful work on each forward pass.

## What It Does

During LLM inference, not all layers contribute equally to every token. Attention patterns are sparse, and activation intensity varies dramatically across the transformer's depth depending on the input. AI Cog Map makes this visible:

- **Layer Activation Heatmap**: A grid where each cell represents a transformer layer. Brightness shows activation intensity — from dark (minimal contribution) through purple (moderate) to bright cyan (heavy activation).
- **Activation Delta Bar**: Shows which layers are *changing* between forward passes — revealing where the model's attention is shifting as it processes new tokens.
- **Live Statistics**: Active layer count, mean activation, data freshness.

Different types of work create different visual signatures. Code generation activates different layers than creative writing. Math reasoning lights up different patterns than summarization. You can literally watch the model's cognitive behavior change.

## Architecture

```
┌─────────────────┐       shared memory       ┌──────────────────┐
│  SGLang Server   │  ──────────────────────>  │  AI Cog Map UI   │
│                  │  /dev/shm/aicogmap-*      │                  │
│  + forward hooks │                           │  FastAPI + HTML   │
│  (aicogmap.hook) │                           │  localhost:7890   │
└─────────────────┘                            └──────────────────┘
```

**Hook Plugin** (`aicogmap.hook`): A SGLang forward hook factory. Attaches to every transformer layer via SGLang's native `--forward-hooks` system. On each forward pass, computes the L2 norm of each layer's output tensor and writes to a shared memory buffer. No SGLang source modification required.

**Visualization Server** (`aicogmap.server`): Standalone FastAPI server that reads the shared memory buffer and serves a real-time heatmap UI. Polls at 500ms intervals for smooth animation.

**Shared Memory Bridge** (`/dev/shm/aicogmap-activations`): Lock-free binary format (4-byte magic + header + float32 array). Zero disk I/O, zero serialization overhead. The hook writes, the server reads.

## Installation

```bash
pip install aicogmap
```

Or from source:

```bash
git clone https://github.com/openhelper/ai-cog-map.git
cd ai-cog-map
pip install -e .
```

## Quick Start

### 1. Start SGLang with hooks enabled

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --dtype bfloat16 \
    --port 8000 \
    --forward-hooks '[{
        "name": "aicogmap",
        "target_modules": ["model.layers.*"],
        "hook_factory": "aicogmap.hook:create_activation_hook",
        "config": {"shm_path": "/dev/shm/aicogmap-activations"}
    }]'
```

### 2. Start the visualization server

```bash
aicogmap --port 7890
```

### 3. Open the UI

Navigate to `http://localhost:7890` and send requests to your model. Watch the layers light up.

## Docker

If SGLang runs in Docker, mount shared memory and the aicogmap package:

```yaml
volumes:
  - /dev/shm:/dev/shm
  - ./aicogmap:/usr/local/lib/python3.12/dist-packages/aicogmap
```

See `examples/docker-compose.example.yml` for a complete setup.

## Configuration

### Hook Config

| Option | Default | Description |
|--------|---------|-------------|
| `shm_path` | `/dev/shm/aicogmap-activations` | Shared memory file path |
| `flush_interval` | `0.1` | Seconds between shared memory flushes |
| `max_layers` | `256` | Maximum layers to track |

### Server Config

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `7890` | HTTP port |
| `--shm-path` | `/dev/shm/aicogmap-activations` | Shared memory path |

## Target Module Patterns

The `target_modules` field uses `fnmatch` patterns. Common patterns:

| Pattern | Matches |
|---------|---------|
| `model.layers.*` | All transformer layers (recommended) |
| `model.layers.*.self_attn` | Attention modules only |
| `model.layers.*.mlp` | Feed-forward modules only |
| `model.layers.0` | First layer only |
| `model.layers.3[0-9]` | Layers 30-39 |

## Performance

- **Hook overhead**: ~0.1-0.5% of inference time (one `tensor.norm()` per layer per forward pass)
- **Memory**: ~1KB shared memory per layer
- **Zero overhead when disabled**: hooks are opt-in via `--forward-hooks`
- **No disk I/O**: shared memory (`/dev/shm`) is RAM-backed

## How It Works

On each forward pass through the model:

1. SGLang's hook manager calls our hook function after each transformer layer executes
2. The hook computes `output.detach().float().norm().item()` — a single scalar capturing the layer's activation magnitude
3. The scalar is written to a thread-safe buffer, indexed by layer position
4. A background thread flushes the buffer to shared memory at configurable intervals
5. The visualization server reads shared memory and normalizes values relative to the current max
6. The frontend renders normalized values as a color-mapped heatmap grid

The L2 norm is chosen because it captures overall activation magnitude without being sensitive to individual outlier values, and it's extremely cheap to compute on GPU tensors.

## Compatibility

- **SGLang**: Any version with `--forward-hooks` support (merged November 2025, PR #13217)
- **Models**: Any transformer-based model served by SGLang
- **GPU**: Any NVIDIA GPU supported by SGLang
- **OS**: Linux (requires `/dev/shm` for shared memory)

## Integration

AI Cog Map exposes a JSON API at `/api/activations` that any dashboard or monitoring tool can consume:

```json
{
  "status": "ok",
  "num_layers": 64,
  "age_s": 0.05,
  "norms": [0.23, 0.45, 0.12, ...],
  "deltas": [0.01, 0.15, 0.03, ...],
  "history": [...]
}
```

Use this to embed cognitive visualization into your own observability stack.

## License

MIT
