"""
This module defines the feature engineering for enriching discrete attributes like
purpose and mode into continuous, meaningful vectors.
"""
from typing import Dict, List

import torch

# --- Enriched Mode Features ---

MODE_FEATURES: Dict[str, List[float]] = {
    "stay":           [0.0, 0.0,  0.0, 1.0],
    "walk":           [1.0, 0.1,  0.0, 0.8],
    "bike":           [1.0, 0.25, 0.1, 0.7],
    "car":            [1.0, 0.7,  0.8, 0.9],
    "public_transit": [1.0, 0.5,  0.4, 0.4],
}
MODE_FEATURE_NAMES = ["is_moving", "avg_speed", "cost_per_km", "convenience"]
MODE_ID_MAP = {name: i for i, name in enumerate(MODE_FEATURES.keys())}
ID_TO_MODE_MAP = {i: name for i, name in enumerate(MODE_FEATURES.keys())}

# --- Enriched Purpose Features ---

PURPOSE_FEATURES: Dict[str, List[float]] = {
    "home":           [1.0, 1.0,  1.0,  0.3],
    "work":           [1.0, 1.0,  0.8,  0.6],
    "education":      [1.0, 0.9,  0.6,  0.7],
    "shopping":       [1.0, 0.2,  0.2,  0.2],
    "social":         [1.0, 0.1,  0.3,  1.0],
    "travel":         [0.0, 0.5,  0.05, 0.0],
}
PURPOSE_FEATURE_NAMES = ["is_stationary", "is_mandatory", "typical_duration", "social_level"]
PURPOSE_ID_MAP = {name: i for i, name in enumerate(PURPOSE_FEATURES.keys())}
ID_TO_PURPOSE_MAP = {i: name for i, name in enumerate(PURPOSE_FEATURES.keys())}


def get_mode_features(mode_id: int) -> torch.Tensor:
    """Returns the feature vector for a given mode ID."""
    mode_name = ID_TO_MODE_MAP.get(mode_id)
    if mode_name is None:
        raise ValueError(f"Invalid mode_id: {mode_id}")
    return torch.tensor(MODE_FEATURES[mode_name], dtype=torch.float32)


def get_purpose_features(purpose_id: int) -> torch.Tensor:
    """Returns the feature vector for a given purpose ID."""
    purpose_name = ID_TO_PURPOSE_MAP.get(purpose_id)
    if purpose_name is None:
        raise ValueError(f"Invalid purpose_id: {purpose_id}")
    return torch.tensor(PURPOSE_FEATURES[purpose_name], dtype=torch.float32)

def get_feature_dimensions():
    """Returns the dimensions of the feature vectors."""
    mode_dim = len(next(iter(MODE_FEATURES.values())))
    purpose_dim = len(next(iter(PURPOSE_FEATURES.values())))
    return mode_dim, purpose_dim

if __name__ == '__main__':
    # Example usage and verification
    mode_dim, purpose_dim = get_feature_dimensions()
    print(f"Mode feature dimension: {mode_dim}")
    print(f"Purpose feature dimension: {purpose_dim}")

    print("\n--- Mode Features ---")
    for name, idx in MODE_ID_MAP.items():
        features = get_mode_features(idx)
        print(f"ID {idx} ({name}):\t{features.numpy()}")

    print("\n--- Purpose Features ---")
    for name, idx in PURPOSE_ID_MAP.items():
        features = get_purpose_features(idx)
        print(f"ID {idx} ({name}):\t{features.numpy()}")
