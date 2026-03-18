"""Custom Transformer + CRF model for token classification."""

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, PretrainedConfig


class TransformerCRF(nn.Module):
    """Wraps any HuggingFace transformer backbone with a CRF head.

    Architecture: AutoModel → dropout → linear → CRF
    Training: CRF negative log-likelihood (replaces cross-entropy)
    Inference: Viterbi decoding (globally optimal label sequence)
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _get_emissions(self, input_ids, attention_mask):
        """Run backbone + classifier to get emission scores."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        return emissions

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass. Returns loss if labels provided, else Viterbi decoded tags.

        Labels of -100 are masked out (special tokens, non-first subwords).
        """
        emissions = self._get_emissions(input_ids, attention_mask)

        if labels is not None:
            # CRF requires the first timestep to be unmasked, so we use
            # attention_mask for the CRF mask (keeps [CLS] and all real tokens).
            # Replace -100 labels with O (0) — CRF can't handle negative indices.
            # These positions ([CLS], [SEP], non-first subwords) get label O
            # which is harmless since the CRF still sees all tokens.
            crf_mask = attention_mask.bool()
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            # CRF returns log-likelihood; negate for loss
            loss = -self.crf(emissions, safe_labels, mask=crf_mask, reduction="mean")
            return loss
        else:
            # Inference: Viterbi decode
            crf_mask = attention_mask.bool()
            tag_seqs = self.crf.decode(emissions, mask=crf_mask)
            return tag_seqs

    def save_pretrained(self, path: str):
        """Save model weights and config info."""
        import json
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        config = {
            "model_name": self.backbone.config._name_or_path,
            "num_labels": self.num_labels,
        }
        (save_dir / "crf_config.json").write_text(json.dumps(config))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        """Load a saved TransformerCRF model."""
        import json
        from pathlib import Path

        save_dir = Path(path)
        config = json.loads((save_dir / "crf_config.json").read_text())
        model = cls(config["model_name"], config["num_labels"])
        state_dict = torch.load(save_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model
