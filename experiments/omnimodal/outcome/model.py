"""Historical candidate-shared scalar scorer; capability training moved to the task classifier."""
import json
from pathlib import Path

import torch
from torch import nn


class SharedQualityModel(nn.Module):
    """Encode each task–candidate pair independently, using left-padded inputs.

    The retained scalar mode starts with zero logits and no numeric prior offset.
    Capability-ID classification has its own model and checkpoint format.
    """

    def __init__(self, encoder, hidden_size, mode="scalar", num_dimensions=None):
        super().__init__()
        if mode not in {"scalar", "capability"}:
            raise ValueError("mode must be scalar or capability")
        if mode == "capability":
            raise ValueError("Candidate capability-quality scoring is retired. Prepare a new task-capability "
                             "plan and use capability_router_20261018/model.py; old capability checkpoints "
                             "remain historical results, not capability-ID classifiers.")
        self.encoder = encoder
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_dimensions = 1
        head_device = next(encoder.parameters()).device
        # Linear's discarded random initialization must not shift the paired
        # recipes' subsequent dropout stream just because their widths differ.
        with torch.random.fork_rng(devices=[head_device.index] if head_device.type == "cuda" else []):
            self.quality_head = nn.Linear(
                hidden_size, self.num_dimensions, dtype=torch.float32, device=head_device,
            )
            nn.init.zeros_(self.quality_head.weight)
            nn.init.zeros_(self.quality_head.bias)

    def forward(self, input_ids, attention_mask, **body_inputs):
        # The PEFT base retains the inserted LoRA modules. Calling its model
        # body avoids allocating vocabulary logits or all-layer hidden states.
        hidden = self.encoder.get_base_model().model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True, **body_inputs,
        ).last_hidden_state[:, -1]
        # BF16 body execution must not downcast the FP32 quality projection.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.quality_head(hidden.float())

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "adapter")
        torch.save(self.quality_head.state_dict(), path / "quality_head.pt")
        (path / "scorer_config.json").write_text(json.dumps({
            "mode": self.mode,
            "hidden_size": self.hidden_size,
            "num_dimensions": self.num_dimensions,
            "head_dtype": "float32",
            "numeric_prior_offset": False,
        }, indent=2) + "\n")

    def load_head(self, path):
        state = torch.load(Path(path) / "quality_head.pt", map_location="cpu", weights_only=True)
        self.quality_head.load_state_dict(state)

    @classmethod
    def load_checkpoint(cls, path, base_model, is_trainable=True, **adapter_kwargs):
        """Restore both adapter and head onto a freshly loaded matching base."""
        from peft import PeftModel

        path = Path(path)
        config = json.loads((path / "scorer_config.json").read_text())
        encoder = PeftModel.from_pretrained(
            base_model, path / "adapter", is_trainable=is_trainable, **adapter_kwargs,
        )
        model = cls(encoder, config["hidden_size"], config["mode"], config["num_dimensions"])
        model.load_head(path)
        return model


def padded_inputs(pairs, tokenizer, device):
    """Left padding keeps the final position equal to the final real token."""
    width = max(map(len, pairs))
    return {
        "input_ids": torch.tensor(
            [[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in pairs], device=device,
        ),
        "attention_mask": torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids in pairs], device=device,
        ),
    }
