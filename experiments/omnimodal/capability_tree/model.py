"""A multimodal task encoder and one global, task-masked capability leaf head."""
import json
from pathlib import Path

import torch
from torch import nn


CHECKPOINT_SCHEMA = "capability-tree-router-v2"
UNRESOLVED_CLASS_ID = "__unresolved__"


class CapabilityTreeClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, leaf_ids):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.leaf_ids = list(leaf_ids)
        self.class_ids = self.leaf_ids + [UNRESOLVED_CLASS_ID]
        self.leaf_to_index = {class_id: index for index, class_id in enumerate(self.class_ids)}
        self.leaf_head = nn.Linear(hidden_size, len(self.class_ids), dtype=torch.float32)
        nn.init.normal_(self.leaf_head.weight, std=0.01)
        nn.init.zeros_(self.leaf_head.bias)

    def forward(self, **inputs):
        # Calling the multimodal backbone avoids materializing vocabulary logits;
        # LoRA modules remain attached to the language layers of this backbone.
        hidden = self.encoder.get_base_model().model(
            **inputs, use_cache=False, return_dict=True,
        ).last_hidden_state
        mask = inputs["attention_mask"]
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.leaf_head(pooled.float())

    def scoped_logits(self, logits, candidate_leaf_ids):
        class_ids = list(candidate_leaf_ids) + [UNRESOLVED_CLASS_ID]
        indexes = torch.tensor([self.leaf_to_index[class_id] for class_id in class_ids],
                               dtype=torch.long, device=logits.device)
        # Restrict CE and argmax to the declared task; no -inf soft-target products.
        return logits.index_select(-1, indexes)

    def save_checkpoint(self, path, metadata=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "adapter")
        torch.save(self.leaf_head.state_dict(), path / "leaf_head.pt")
        (path / "classifier_config.json").write_text(json.dumps({
            "schema_version": CHECKPOINT_SCHEMA,
            "hidden_size": self.hidden_size,
            "leaf_ids": self.leaf_ids,
            "class_ids": self.class_ids,
            "unresolved_class_id": UNRESOLVED_CLASS_ID,
            "metadata": metadata or {},
        }, ensure_ascii=False, indent=2) + "\n")

    def restore_weights(self, path):
        """Reload the best epoch without allocating another base model on GPU."""
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        path = Path(path)
        state = load_peft_weights(str(path / "adapter"), device="cpu")
        set_peft_model_state_dict(self.encoder, state)
        self.leaf_head.load_state_dict(torch.load(path / "leaf_head.pt", map_location="cpu", weights_only=True))

    @classmethod
    def load_checkpoint(cls, path, base_model, leaf_ids, is_trainable=True):
        from peft import PeftModel

        path = Path(path)
        config = json.loads((path / "classifier_config.json").read_text())
        if config["schema_version"] != CHECKPOINT_SCHEMA or config["leaf_ids"] != list(leaf_ids):
            raise ValueError("Checkpoint and data must use the same capability tree and leaf order.")
        encoder = PeftModel.from_pretrained(base_model, path / "adapter", is_trainable=is_trainable)
        classifier = cls(encoder, config["hidden_size"], leaf_ids)
        classifier.leaf_head.load_state_dict(torch.load(path / "leaf_head.pt", map_location="cpu", weights_only=True))
        return classifier


def load_model(model_path, leaf_ids, device="cuda", init_checkpoint=None, lora_r=8):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa", local_files_only=False,
    )
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    if init_checkpoint:
        model = CapabilityTreeClassifier.load_checkpoint(init_checkpoint, base, leaf_ids)
    else:
        targets = [name for name, module in base.named_modules()
                   if isinstance(module, nn.Linear) and "language_model.layers." in name]
        encoder = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
            target_modules=targets, task_type="CAUSAL_LM", bias="none",
        ))
        model = CapabilityTreeClassifier(encoder, base.config.text_config.hidden_size, leaf_ids)
    model.to(device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.encoder.enable_input_require_grads()
    return model
