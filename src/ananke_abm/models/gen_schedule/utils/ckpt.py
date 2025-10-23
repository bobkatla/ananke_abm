
import os
import torch
def save_checkpoint(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
