import numpy as np

def muvera_score(q: np.ndarray, D: np.ndarray) -> float:
    scores = D @ q
    weights = np.exp(scores - np.max(scores))
    weights /= np.sum(weights)
    return float(np.sum(weights * scores))
