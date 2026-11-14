import torch
import torch.nn as nn

FEATURE_COLUMNS = [
    "creature_mass_talents", "oracle_omen", "aggression_index", "divine_aura",
    "aegis_resonance", "armor_resistance", "witness_count",
    "styx_disturbance", "omen_intensity"
]

class MythicThreatClassifier(nn.Module):
    """Tiny binary classifier shared by all three submitted checkpoints."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(FEATURE_COLUMNS), 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
