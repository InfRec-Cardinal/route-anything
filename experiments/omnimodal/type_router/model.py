"""One task encoding predicts a capability ID; model selection is a table lookup."""
import json
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_SCHEMA = 'task-capability-classifier-v1'


class CapabilityClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, dimensions):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.dimensions = dimensions
        device = next(encoder.parameters()).device
        self.capability_head = nn.Linear(hidden_size, len(dimensions), device=device, dtype=torch.float32)
        nn.init.zeros_(self.capability_head.weight)
        nn.init.zeros_(self.capability_head.bias)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder.get_base_model().model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).last_hidden_state[:, -1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.capability_head(hidden.float())

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / 'adapter')
        torch.save(self.capability_head.state_dict(), path / 'capability_head.pt')
        (path / 'classifier_config.json').write_text(json.dumps({
            'schema_version': CHECKPOINT_SCHEMA,
            'hidden_size': self.hidden_size,
            'dimensions': self.dimensions,
        }, ensure_ascii=False, indent=2) + '\n')

    def load_head(self, path):
        self.capability_head.load_state_dict(torch.load(
            Path(path) / 'capability_head.pt', map_location='cpu', weights_only=True))

    @classmethod
    def load_checkpoint(cls, path, base_model, is_trainable=False):
        from peft import PeftModel

        path = Path(path)
        config_path = path / 'classifier_config.json'
        if not config_path.exists():
            raise ValueError('This is not a task-capability checkpoint. Rebuild the new data plan and train '
                             'a task classifier; the old candidate-quality head cannot be reused.')
        config = json.loads(config_path.read_text())
        if config['schema_version'] != CHECKPOINT_SCHEMA:
            raise ValueError('Unsupported capability checkpoint schema; rebuild with the new trainer.')
        encoder = PeftModel.from_pretrained(base_model, path / 'adapter', is_trainable=is_trainable)
        model = cls(encoder, config['hidden_size'], config['dimensions'])
        model.load_head(path)
        return model


def padded_inputs(encoded, pad_token_id, device):
    width = max(map(len, encoded))
    return {
        'input_ids': torch.tensor([[pad_token_id] * (width-len(ids)) + ids for ids in encoded], device=device),
        'attention_mask': torch.tensor([[0] * (width-len(ids)) + [1] * len(ids) for ids in encoded], device=device),
    }
