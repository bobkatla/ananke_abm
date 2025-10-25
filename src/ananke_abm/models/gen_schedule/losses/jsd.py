from scipy.spatial.distance import jensenshannon
import numpy as np

def jsd(p, q, eps=1e-12):
    # scipy expects non-negative vectors; they'll get normalized internally.
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # tiny epsilon to avoid all-zeros corner cases
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    js_distance = jensenshannon(p, q, base=np.e)  # base=np.e so logs match our nats
    js_divergence = js_distance ** 2
    return float(js_divergence)
