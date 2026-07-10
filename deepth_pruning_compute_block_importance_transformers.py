import torch
from transformers import AutoTokenizer,Qwen2ForCausalLM
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to("cuda")

def compute_block_importance(input_hidden_states,hidden_states):
    return  F.cosine_similarity(input_hidden_states[0], hidden_states[0], dim=-1).mean()

def attach_block_importance(model, block_importance: dict):
    """
    Registers a forward hook on every Qwen2DecoderLayer that accumulates
    a simple scalar importance score into block_importance[layer_idx].
    block_importance should be passed in as {} and gets filled in-place.
    """
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            
            score = compute_block_importance(input_hidden_states=inputs,hidden_states= output)
            # ----------------------------------------------------------
            if layer_idx not in block_importance:
                block_importance[layer_idx] = [score]
            else:
                block_importance[layer_idx].append(score)

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    return handles

block_importance = {}
handles = attach_block_importance(model, block_importance)


prompts = ["write a python code for file search in all sub directrioes ?",
           "tell me about you",
           "explain capitalism vs sociallism what do you prefer"]

for prompt in prompts:
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=500)

for h in handles:
    h.remove()


block_importance = {
    k: torch.stack(v).cpu() for k, v in block_importance.items()
}
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


for key in block_importance:
    print(key,1 - block_importance[key].mean())
