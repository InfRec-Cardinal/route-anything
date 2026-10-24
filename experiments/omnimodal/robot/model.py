"""Shared Qwen multimodal encoder with seven independent success logits."""
import json
from pathlib import Path

import torch
from torch import nn


class RouterClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, model_ids):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.model_ids = list(model_ids)
        self.head = nn.Linear(hidden_size, len(self.model_ids), dtype=torch.float32)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, **inputs):
        hidden = self.encoder.get_base_model().model(
            **inputs, use_cache=False, return_dict=True,
        ).last_hidden_state
        mask = inputs["attention_mask"]
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(pooled.float())

    def save_checkpoint(self, path, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "adapter")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "router_config.json").write_text(json.dumps({
            "schema_version": "robotwin-router-success-v1",
            "hidden_size": self.hidden_size,
            "model_ids": self.model_ids,
            "metadata": metadata,
        }, ensure_ascii=False, indent=2) + "\n")

    def restore_weights(self, path):
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        path = Path(path)
        config = json.loads((path / "router_config.json").read_text())
        if config["model_ids"] != self.model_ids:
            raise ValueError("Checkpoint model order differs from the current data.")
        state = load_peft_weights(str(path / "adapter"), device="cpu")
        set_peft_model_state_dict(self.encoder, state)
        self.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))


def load_model(model_path, model_ids, device, lora_r=8):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa", local_files_only=False,
    )
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    targets = [name for name, module in base.named_modules()
               if isinstance(module, nn.Linear) and "language_model.layers." in name]
    encoder = get_peft_model(base, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
        target_modules=targets, task_type="CAUSAL_LM", bias="none",
    ))
    model = RouterClassifier(encoder, base.config.text_config.hidden_size, model_ids).to(device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.encoder.enable_input_require_grads()
    return model
