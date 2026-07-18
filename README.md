# LLM Pruning: Width + Depth

This module is a paper implementation of [*Compact Language Models via Pruning and Knowledge Distillation*](https://arxiv.org/pdf/2407.14679).

The code is my best effort at a generalized pruning implementation for Llama/Mistral/Qwen2-style causal LMs. I'd appreciate any help finding bugs or suggesting improvements.

- **Zero-shot, no retraining:** use depth pruning, up to ~25% of layers.
- **Best accuracy recovery:** use width pruning, followed by retraining/distillation.

## Files

| File | What it prunes | Needs retraining? |
|---|---|---|
| `width_pruning.py` | Attention heads, hidden-dim channels, MLP intermediate channels | Recommended for best accuracy recovery |
| `depth_pruning.py` | Whole transformer decoder layers | Optional; usable zero-shot up to ~25% |

Both scripts follow the same three-stage pattern from the paper: attach importance-scoring hooks → run calibration data through the model → use the collected scores to decide what to keep, then physically resize/remove the corresponding modules.

## `width_pruning.py`

Prunes the model's width along three axes, all derived from the same calibration pass:

1. **Attention heads** — importance scored from `o_proj`'s input, grouped by KV group (GQA-aware, so pruning never breaks the query/KV head mapping).
2. **Hidden dimension** — the *count* of channels kept is derived from how many heads survive (`num_heads_kept * head_dim`); *which* channels survive is chosen from importance scores accumulated at every RMSNorm in the model (each layer's `input_layernorm` / `post_attention_layernorm`, plus the final `model.model.norm`).
3. **MLP intermediate width** — scored independently per layer from `gate_proj`/`up_proj` output activations.

`keep_ratio` is the fraction of attention heads to keep (ceil-rounded per KV group, e.g. 0.75 of 28 heads → 21). Everything else (hidden size, per-layer MLP width) is derived or matched to that.

```python
from width_pruning import run_width_pruning
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16")

calibration_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, model compression reduces size and latency.",
]

run_width_pruning(
    model, tokenizer, calibration_texts,
    keep_ratio=0.75,
    save_dir="./pruned_model",
    device="cuda",
)
```

Width-pruned models lose accuracy immediately (every projection matrix is physically resized) and are expected to need retraining/distillation to recover it — this is the path the paper recommends for the best final accuracy.

## `depth_pruning.py`

Scores each decoder layer by how much it transforms its input: `1 - cosine_similarity(block_input, block_output)`. A score near 0 means the layer is close to an identity function (safe to drop); a higher score means the layer does real representational work (keep it).

```python
from depth_pruning import run_depth_pruning
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to("cuda")

calibration_prompts = [
    "write a python function for file search in all subdirectories",
    "tell me about yourself",
    "explain capitalism vs socialism, what do you prefer",
]

run_depth_pruning(
    model, tokenizer, calibration_prompts,
    prune_ratio=0.25,        # or num_layers_to_prune=<int>
    max_new_tokens=200,
    save_dir="./depth_pruned_model",
    device="cuda",
)
```

Removing whole layers requires no weight surgery — it's a structurally intact model with fewer blocks — which is why it holds up reasonably well zero-shot. Two details matter for correctness on save/generate:

- **`layer_idx` re-indexing.** HF's KV cache (`DynamicCache`) addresses storage by each layer's `layer_idx`. If pruned layers leave gaps (e.g. layers 3, 7, 12 survive out of 24), the cache will write to the wrong slots the first time you generate with caching on. `prune_depth` renumbers survivors to `0..N-1` in forward-pass order.
- **Per-layer config fields.** Some newer configs (e.g. Qwen2/Qwen3's `layer_types`, a per-layer sliding-window/full-attention pattern) must have exactly `num_hidden_layers` entries or `config.validate()` rejects the save. `prune_depth` filters any config list/tuple whose length matches the original layer count down to the surviving indices, so it stays aligned with the pruned layers.

By default, importance is scored only on the prefill call per prompt (`prefill_only=True`), not on every autoregressive decode step — otherwise hundreds of single-token decode-step scores would dominate the signal over the one call that actually saw the full prompt.

## Assumptions

Both scripts assume a Llama/Mistral/Qwen2-style architecture:

```
model.model.embed_tokens
model.model.layers[i].self_attn.{q,k,v,o}_proj
model.model.layers[i].mlp.{gate,up,down}_proj
model.model.layers[i].{input_layernorm,post_attention_layernorm}
model.model.norm
model.lm_head
```

RMSNorm modules are assumed to expose only `.weight` (no bias). Other architectures (e.g. models with QKV bias, different norm placement, or non-GQA attention) haven't been tested and likely need adjustment.

## Known limitations / open questions

- No retraining or distillation step is included here — that's a separate stage per the paper, not automated by these scripts.
- Width pruning currently reuses the same `keep_ratio` across heads, hidden dim, and MLP width; the paper explores independently tuning these per axis, which isn't exposed here yet.
- Importance scores are accumulated as a running sum over calibration batches rather than normalized per batch, so scores aren't directly comparable across runs with different amounts of calibration data.
- Depth-pruning's `prefill_only` heuristic (see above) is a judgment call, not something validated against the paper's own layer-importance methodology — worth checking against their reported metric if exact reproduction matters.

Bug reports and PRs welcome.
