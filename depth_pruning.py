"""
Depth pruning for Qwen2/Llama-style HF causal LMs.

Pipeline
--------
1. attach_block_importance(...)  register forward hooks on every decoder
                                  layer that score how much that layer
                                  transforms its input (1 - cosine sim
                                  between block input and block output).
                                  Low score  -> near-identity layer -> prunable.
                                  High score -> layer does real work -> keep.
2. Run calibration prompts through the model.
3. aggregate_block_importance(...) collapse the per-call score lists into
                                    one scalar per layer.
4. select_layers_to_prune(...)     pick the N lowest-importance layers.
5. prune_depth(...)                physically delete those decoder layers,
                                    re-index the survivors' layer_idx
                                    (needed for KV-cache addressing), and
                                    update model.config.num_hidden_layers.

Assumes model.model.layers is an nn.ModuleList of decoder layers, and that
each layer (and its self_attn submodule) stores a `layer_idx` attribute used
to address the KV cache -- true for Qwen2, Llama, Mistral, etc.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1. Importance collection
# --------------------------------------------------------------------------- #

def compute_block_importance(input_hidden_states: torch.Tensor,
                              output_hidden_states: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between a block's input and output hidden
    states. ~0 means the block barely changed anything (near-identity,
    prunable); closer to 1 means the block transformed the representation a
    lot (load-bearing, keep)."""
    cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)
    return (1 - cos_sim).mean()


def attach_block_importance(model, block_importance: dict, prefill_only: bool = True):
    """
    Registers a forward hook on every decoder layer that accumulates a
    scalar importance score into block_importance[layer_idx] (list, one
    entry per hook firing). block_importance should be passed in as {} and
    is filled in-place.

    prefill_only: with KV caching, model.generate() calls each layer once
    per decode step (seq_len == 1) in addition to the initial prefill call
    (seq_len == prompt length). If True (default), only the first call per
    generate() is scored, so the signal reflects how each layer treats the
    actual prompt rather than being swamped by hundreds of single-token
    decode-step scores. Set False to score every call (original behavior).
    """
    handles = []
    calls_seen = {}  # layer_idx -> int, reset externally between prompts if desired

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden_in = inputs[0] if isinstance(inputs, tuple) else inputs
            hidden_out = output[0] if isinstance(output, tuple) else output

            if prefill_only:
                # only score calls that see more than one token (the prefill)
                if hidden_in.shape[1] <= 1:
                    return

            score = compute_block_importance(hidden_in, hidden_out)
            block_importance.setdefault(layer_idx, []).append(score.detach().cpu())
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    return handles


def aggregate_block_importance(block_importance: dict) -> dict:
    """Collapse {layer_idx: [scalar, scalar, ...]} into {layer_idx: float}."""
    return {
        layer_idx: torch.stack(scores).mean().item()
        for layer_idx, scores in block_importance.items()
    }


# --------------------------------------------------------------------------- #
# 2. Keep/prune selection
# --------------------------------------------------------------------------- #

def select_layers_to_prune(importance: dict, num_layers_to_prune: int = None,
                            prune_ratio: float = None) -> list:
    """
    importance: {layer_idx: float}, higher = more important (keep).
    Specify exactly one of num_layers_to_prune or prune_ratio.
    Returns a sorted list of layer indices to remove (lowest importance first).
    """
    assert (num_layers_to_prune is None) != (prune_ratio is None), \
        "specify exactly one of num_layers_to_prune / prune_ratio"

    total = len(importance)
    if prune_ratio is not None:
        num_layers_to_prune = round(total * prune_ratio)
    num_layers_to_prune = max(0, min(total - 1, num_layers_to_prune))  # never prune every layer

    ranked = sorted(importance.items(), key=lambda kv: kv[1])  # ascending: least important first
    prune_idx = [layer_idx for layer_idx, _ in ranked[:num_layers_to_prune]]
    return sorted(prune_idx)


# --------------------------------------------------------------------------- #
# 3. Depth surgery
# --------------------------------------------------------------------------- #

def prune_depth(model, prune_idx: list):
    """
    Physically removes the decoder layers listed in prune_idx, re-indexes
    the survivors' layer_idx (both on the decoder layer and its self_attn
    submodule, since that index addresses the KV cache), updates
    model.config.num_hidden_layers, and slices any other per-layer config
    fields (e.g. `layer_types`, which some newer HF configs -- Qwen2/Qwen3 --
    use to record a sliding-window vs full-attention pattern per layer) down
    to match.

    Re-indexing matters: HF's Cache implementations (e.g. DynamicCache)
    store one KV entry per layer_idx. If the surviving layers keep their
    original, now-noncontiguous layer_idx values, cache writes/reads will
    go to the wrong slots (or out of range) the moment generation uses
    caching. This walks the new ModuleList in order and reassigns
    layer_idx = 0..N-1 to match actual forward-pass order.

    Per-layer config fields matter too: config.validate() (run inside
    save_pretrained) requires len(config.layer_types) == num_hidden_layers.
    Any config attribute that happens to be a list/tuple with one entry per
    original layer is filtered the same way, keeping the same relative
    order as the surviving layers, so it stays aligned with them.
    """
    prune_set = set(prune_idx)
    original_num_layers = len(model.model.layers)
    keep_mask = [i not in prune_set for i in range(original_num_layers)]

    kept_layers = [layer for i, layer in enumerate(model.model.layers) if keep_mask[i]]

    for new_idx, layer in enumerate(kept_layers):
        layer.layer_idx = new_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx

    model.model.layers = nn.ModuleList(kept_layers)
    model.config.num_hidden_layers = len(kept_layers)

    # Slice any other per-layer config list/tuple (layer_types, and
    # anything similar future/other architectures might add) to match.
    for attr, value in list(vars(model.config).items()):
        if isinstance(value, (list, tuple)) and len(value) == original_num_layers:
            filtered = type(value)(v for v, keep in zip(value, keep_mask) if keep)
            setattr(model.config, attr, filtered)

    return model


# --------------------------------------------------------------------------- #
# 4. Orchestration
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_depth_pruning(model, tokenizer, calibration_prompts,
                       num_layers_to_prune: int = None, prune_ratio: float = None,
                       max_new_tokens: int = 200, save_dir: str = "./depth_pruned_model",
                       device: str = "cuda"):
    model.eval().to(device)

    block_importance = {}
    handles = attach_block_importance(model, block_importance, prefill_only=True)

    for prompt in calibration_prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        model.generate(**inputs, max_new_tokens=max_new_tokens)

    for h in handles:
        h.remove()

    importance = aggregate_block_importance(block_importance)
    prune_idx = select_layers_to_prune(importance, num_layers_to_prune, prune_ratio)

    print(f"Pruning {len(prune_idx)}/{len(importance)} layers: {prune_idx}")
    for layer_idx in sorted(importance):
        flag = " <-- PRUNED" if layer_idx in prune_idx else ""
        print(f"  layer {layer_idx:3d}  score={importance[layer_idx]:.4f}{flag}")

    prune_depth(model, prune_idx)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, importance, prune_idx


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to("cuda")

    calibration_prompts = [
        "write a python code for file search in all sub directrioes ?",
        "tell me about you",
        "explain capitalism vs sociallism what do you prefer",
    ]

    run_depth_pruning(
        model, tokenizer, calibration_prompts,
        prune_ratio=0.25,          # or num_layers_to_prune=7
        max_new_tokens=200,
        save_dir="./depth_pruned_model",
        device="cuda",
    )
