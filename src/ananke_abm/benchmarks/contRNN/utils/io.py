import os
import torch

def load_encoded(path_pt: str):
    if os.path.exists(path_pt):
        obj = torch.load(path_pt, map_location="cpu")
        return obj
    raise FileNotFoundError(path_pt)

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def save_ckpt(path, model, optim, epoch, best=False):
    obj = {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "best": best}
    torch.save(obj, path)

def load_vocab(path):
    import json
    return json.load(open(path))
