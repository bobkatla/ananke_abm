import numpy as np

def _safe_norm(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0: 
        return np.ones_like(p)/len(p)
    return p / s

def jsd(p, q, eps=1e-12):
    p = _safe_norm(p)
    q = _safe_norm(q)
    m = 0.5*(p+q)
    def ent(x):
        x = np.clip(x, eps, 1.0)
        return -np.sum(x*np.log(x))
    return 0.5*(ent(p) - ent(m)) + 0.5*(ent(q) - ent(m))
