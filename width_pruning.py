"""
Width pruning for Llama-style HF causal LMs.

Pipeline
--------
1. attach_importance_hooks(...)   register hooks that accumulate importance
                                   scores for hidden-dim channels, MLP
                                   intermediate channels, and attention heads.
2. Run a handful of calibration batches through the model in eval/no_grad mode.
3. run_width_pruning(...)         picks heads to keep, derives the hidden
                                   size from that, picks MLP channels, then
                                   physically resizes every affected
                                   weight/bias/norm tensor, updates
                                   model.config, and saves the result.

Assumes a Llama/Mistral/Qwen2-style architecture:
    model.model.embed_tokens
    model.model.layers[i].self_attn.{q,k,v,o}_proj
    model.model.layers[i].mlp.{gate,up,down}_proj
    model.model.norm
    model.lm_head
RMSNorm modules are assumed to expose only `.weight` (no bias).

keep_ratio semantics
---------------------
`keep_ratio` is the fraction of *attention heads* to keep (ceil-rounded, so
0.75 of 28 heads -> 21 heads, not 20). The pruned hidden_size is DERIVED from
that head count (num_heads_kept * head_dim) rather than picked independently.
Which specific hidden-dim channels survive (for embed_tokens / lm_head / all
norm weights) is decided separately, by the "channel" importance scores
accumulated from every RMSNorm in the model (input_layernorm,
post_attention_layernorm, and the final norm). MLP intermediate channels are
still selected by their own importance scores, using the same ceil-rounded
keep count.
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1. Importance collection
# --------------------------------------------------------------------------- #

def _accumulate(store: dict, key: str, scores: torch.Tensor):
    """Running sum of |scores|, so contributions never cancel out."""
    scores = scores.detach().float().abs().cpu()
    store[key] = store.get(key, 0.0) + scores


def attach_importance_hooks(model, num_heads: int, num_kv_heads: int, head_dim: int):
    """
    Registers forward hooks that accumulate importance scores in-place.

    Returns (handles, importance):
      importance["channel"]             -> (hidden_size,)   from EVERY norm:
                                            each layer's input_layernorm and
                                            post_attention_layernorm, plus the
                                            final model.model.norm.
      importance[f"layer_{i}.mlp"]      -> (intermediate_size,)
      importance[f"layer_{i}.self_attn"] -> (num_kv_heads,)  [grouped]

    Call model.eval(), run calibration data through it, then remove the hooks
    and read the dicts.
    """
    handles = []
    importance: dict = {}

    # -- channel importance, accumulated from every RMSNorm in the model --- #
    # NOTE: abs() is taken per-token (before the sum over batch/seq) so that
    # a channel firing strongly positive on some tokens and strongly negative
    # on others doesn't cancel to ~0 and look unimportant. Reducing first and
    # taking abs() afterwards (the previous behavior) only prevents
    # cancellation *across calibration batches*, not across tokens within a
    # batch, which is where most of the cancellation risk actually lives.
    def channel_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        scores = hidden_states.abs().sum(dim=(0, 1))  # (hidden_size,)
        _accumulate(importance, "channel", scores)

    group_size = num_heads // num_kv_heads

    for i, layer in enumerate(model.model.layers):

        # -- attention head importance, from o_proj's input ---------------- #
        # o_proj's input is always laid out by *query* heads (num_heads),
        # even under GQA. We collapse query heads into their KV group so the
        # score reflects the shared K/V head they depend on.
        def make_attn_hook(layer_idx):
            def hook(module, inputs):
                X = inputs[0]  # (batch, seq, num_heads * head_dim)
                b, s, _ = X.shape
                X = X.view(b, s, num_heads, head_dim)
                per_query_head = X.abs().sum(dim=(0, 1, 3))  # (num_heads,)
                per_kv_group = per_query_head.view(num_kv_heads, group_size).sum(dim=1)
                _accumulate(importance, f"layer_{layer_idx}.self_attn", per_kv_group)
            return hook

        def make_channel_hook(layer_idx):
            def channel_hook(module, inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                scores = hidden_states.abs().sum(dim=(0, 1))  # (hidden_size,)
                _accumulate(importance, "channel", scores)
            return channel_hook

        # both norms in this layer feed the channel importance score
        # NOTE: must register the *instantiated* hook (i.e. call the
        # factory), not the factory function itself. Registering
        # `make_channel_hook` directly makes PyTorch call it as
        # hook(module, input, output) -- but the factory only accepts one
        # positional arg (layer_idx) -- raising a TypeError on the first
        # forward pass. Also, forward hooks are always invoked as
        # (module, input, output), so the inner function must accept all
        # three params even when `input` is unused.
        handles.append(layer.input_layernorm.register_forward_hook(make_channel_hook(i)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(make_channel_hook(i)))

        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            make_attn_hook(i)))

        # -- MLP intermediate-channel importance (gate_proj & up_proj) ----- #
        def make_mlp_hook(layer_idx):
            def hook(module, inputs, output):
                scores = output.abs().sum(dim=(0, 1))  # (intermediate_size,)
                _accumulate(importance, f"layer_{layer_idx}.mlp", scores)
            return hook

        handles.append(layer.mlp.gate_proj.register_forward_hook(make_mlp_hook(i)))
        handles.append(layer.mlp.up_proj.register_forward_hook(make_mlp_hook(i)))

    # final norm before lm_head also feeds channel importance
    handles.append(model.model.norm.register_forward_hook(channel_hook))

    return handles, importance


# --------------------------------------------------------------------------- #
# 2. Keep-index selection
# --------------------------------------------------------------------------- #

def _keep_k(total: int, keep_ratio: float) -> int:
    """How many of `total` items to keep, ceil-rounded (0.75 * 28 -> 21)."""
    return max(1, min(total, math.ceil(total * keep_ratio)))


def select_keep_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k indices by score. `k` is resolved ahead of time by the caller
    (via `_keep_k`, or derived from head pruning for the hidden dim)."""
    k = max(1, min(k, len(scores)))
    return torch.topk(scores, k).indices.sort().values


def select_keep_heads_gqa(group_scores: torch.Tensor, num_heads: int,
                           num_kv_heads: int, keep_ratio: float):
    """group_scores has one entry per KV group (see attach_importance_hooks).
    keep_ratio is the fraction of heads to keep, ceil-rounded per KV group so
    every kept query head still has an intact K/V group to attend to."""
    group_size = num_heads // num_kv_heads
    k_groups = _keep_k(num_kv_heads, keep_ratio)
    keep_kv = torch.topk(group_scores, k_groups).indices.sort().values
    keep_query = torch.cat([torch.arange(g * group_size, (g + 1) * group_size) for g in keep_kv])
    return keep_query, keep_kv


# --------------------------------------------------------------------------- #
# 3. Weight surgery
# --------------------------------------------------------------------------- #

def _slice_in(lin: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    """New Linear with in_features restricted to idx."""
    new_lin = nn.Linear(len(idx), lin.out_features, bias=lin.bias is not None,
                         device=lin.weight.device, dtype=lin.weight.dtype)
    new_lin.weight.data = lin.weight.data[:, idx].clone()
    if lin.bias is not None:
        new_lin.bias.data = lin.bias.data.clone()
    return new_lin


def _slice_out(lin: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    """New Linear with out_features restricted to idx."""
    new_lin = nn.Linear(lin.in_features, len(idx), bias=lin.bias is not None,
                         device=lin.weight.device, dtype=lin.weight.dtype)
    new_lin.weight.data = lin.weight.data[idx, :].clone()
    if lin.bias is not None:
        new_lin.bias.data = lin.bias.data[idx].clone()
    return new_lin


def prune_hidden_dim(model, keep_idx: torch.Tensor):
    """Prunes the residual-stream width. Touches embed_tokens, lm_head, every
    layer's q/k/v_proj (input), o_proj (output), gate/up_proj (input),
    down_proj (output), both layernorms, and the final norm."""
    # 1. Prune embed_tokens
    emb = model.model.embed_tokens
    new_emb = nn.Embedding(emb.num_embeddings, len(keep_idx),
                            device=emb.weight.device, dtype=emb.weight.dtype)
    new_emb.weight.data = emb.weight.data[:, keep_idx].clone()
    model.model.embed_tokens = new_emb

    # 2. Prune lm_head (input dimension matches the embedding channel)
    model.lm_head = _slice_in(model.lm_head, keep_idx)

    # 3. Prune final LayerNorm
    model.model.norm.weight.data = model.model.norm.weight.data[keep_idx].clone()
    # Note: if your norm has a bias, slice that too:
    # model.model.norm.bias.data = model.model.norm.bias.data[keep_idx].clone()

    # 4. Iterate through every transformer block to prune shared embedding dims
    for layer in model.model.layers:
        # A. Prune LayerNorms
        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[keep_idx].clone()
        layer.post_attention_layernorm.weight.data = \
            layer.post_attention_layernorm.weight.data[keep_idx].clone()

        # B. Prune MHA projection layers
        # The input to Q, K, V is the embedding dimension
        layer.self_attn.q_proj = _slice_in(layer.self_attn.q_proj, keep_idx)
        layer.self_attn.k_proj = _slice_in(layer.self_attn.k_proj, keep_idx)
        layer.self_attn.v_proj = _slice_in(layer.self_attn.v_proj, keep_idx)
        # The output of O is the embedding dimension
        layer.self_attn.o_proj = _slice_out(layer.self_attn.o_proj, keep_idx)

        # C. Prune MLP projection layers (embedding-dimension side only)
        # The input to gate and up is the embedding dimension
        layer.mlp.gate_proj = _slice_in(layer.mlp.gate_proj, keep_idx)
        layer.mlp.up_proj = _slice_in(layer.mlp.up_proj, keep_idx)
        # The output of down is the embedding dimension
        layer.mlp.down_proj = _slice_out(layer.mlp.down_proj, keep_idx)


def prune_mlp_layer(layer, keep_idx: torch.Tensor):
    """Prunes one layer's MLP intermediate width (gate/up out, down in)."""
    mlp = layer.mlp
    mlp.gate_proj = _slice_out(mlp.gate_proj, keep_idx)
    mlp.up_proj = _slice_out(mlp.up_proj, keep_idx)
    mlp.down_proj = _slice_in(mlp.down_proj, keep_idx)


def prune_attention_heads(layer, keep_query_heads: torch.Tensor,
                           keep_kv_heads: torch.Tensor, head_dim: int):
    """Prunes one layer's attention heads (GQA-aware)."""
    attn = layer.self_attn

    def head_idx(heads):
        return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads])

    attn.q_proj = _slice_out(attn.q_proj, head_idx(keep_query_heads))
    attn.k_proj = _slice_out(attn.k_proj, head_idx(keep_kv_heads))
    attn.v_proj = _slice_out(attn.v_proj, head_idx(keep_kv_heads))
    attn.o_proj = _slice_in(attn.o_proj, head_idx(keep_query_heads))

    attn.num_heads = len(keep_query_heads)
    attn.num_key_value_heads = len(keep_kv_heads)
    if hasattr(attn, "hidden_size"):
        attn.hidden_size = len(keep_query_heads) * head_dim


# --------------------------------------------------------------------------- #
# 4. Orchestration
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_width_pruning(model, tokenizer, calibration_texts, keep_ratio: float = 0.5,
                       save_dir: str = "./pruned_model", device: str = "cuda"):
    """
    Runs calibration, computes importance, then prunes:
      - heads first: keep_ratio is the fraction of heads to keep (ceil-rounded)
      - hidden dim: SIZE is derived from heads kept (num_heads_kept * head_dim),
        but WHICH channels survive is chosen from channel importance scores
      - MLP intermediate width: chosen from each layer's own MLP importance,
        count ceil-rounded the same way as heads
    then saves the pruned model.
    """

    model.eval().to(device)
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = cfg.hidden_size // num_heads

    handles, importance = attach_importance_hooks(model, num_heads, num_kv_heads, head_dim)

    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        model(**inputs)

    for h in handles:
        h.remove()

    # -- heads: keep_ratio is the fraction of heads to keep, ceil-rounded -- #
    group_size = num_heads // num_kv_heads
    num_kv_groups_kept = _keep_k(num_kv_heads, keep_ratio)
    num_heads_kept = num_kv_groups_kept * group_size

    # -- hidden dim: count comes from heads kept, indices from channel
    #    importance (accumulated from every norm in the model) ------------- #
    new_hidden_size = num_heads_kept * head_dim
    keep_hidden = select_keep_indices(importance["channel"], new_hidden_size)
    prune_hidden_dim(model, keep_hidden)

    new_intermediate_size = None

    for i, layer in enumerate(model.model.layers):
        mlp_scores = importance[f"layer_{i}.mlp"]
        keep_mlp = select_keep_indices(mlp_scores, _keep_k(len(mlp_scores), keep_ratio))
        prune_mlp_layer(layer, keep_mlp)
        new_intermediate_size = len(keep_mlp)

        keep_q, keep_kv = select_keep_heads_gqa(
            importance[f"layer_{i}.self_attn"], num_heads, num_kv_heads, keep_ratio)
        prune_attention_heads(layer, keep_q, keep_kv, head_dim)

    cfg.hidden_size = new_hidden_size
    cfg.intermediate_size = new_intermediate_size
    cfg.num_attention_heads = num_heads_kept
    cfg.num_key_value_heads = num_kv_groups_kept

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

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
    
