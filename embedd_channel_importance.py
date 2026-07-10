import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to("cuda")


def compute_channel_importance(X: torch.Tensor, channel_importance: dict, layer_name: str):
    hidden_dim = X.shape[-1]
    X = X.view(-1, hidden_dim)
    col_sums = X.sum(dim=0).detach().cpu()
    channel_importance.setdefault(layer_name, []).append(col_sums)


def attach_channel_importance(model, channel_importance: dict):
    handles = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            compute_channel_importance(hidden_states, channel_importance, layer_name)
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
    return handles


channel_importance = {}
handles = attach_channel_importance(model, channel_importance)

prompts = [
    "write a python code for file search in all sub directrioes ?",
    "tell me about you",
    "explain capitalism vs sociallism what do you prefer",
]

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500)

for h in handles:
    h.remove()

channel_importance = {k: torch.stack(v) for k, v in channel_importance.items()}

for k, v in channel_importance.items():
    print(k, v.shape)
    print(v.sum(0))
    