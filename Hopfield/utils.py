from numbers import Number
from typing import  Optional
import numpy as np


__all__ = [
    "shuffle_degrade",
    "noise_degrade",
    "sgn",
    "squash",
    "zero_center",
]


def shuffle_degrade(X: np.ndarray, scale: float) -> np.ndarray:

    out = X.copy()

    if X.ndim == 1:
        N = len(X)
        ix_changing = np.random.choice(N, round(scale * N))
        out[ix_changing] = X[np.random.choice(N, len(ix_changing))]
        return out

    elif X.ndim == 2:
        N = X.shape[1]
        for i in range(X.shape[0]):
            ix_changing = np.random.choice(N, round(scale * N))
            out[i][ix_changing] = X[i][np.random.choice(N, len(ix_changing))]
        return out

    raise ValueError("input must be 1D or 2D")


def noise_degrade(arr: np.ndarray,
                  scale: Optional[float] = 0.1,
                  loc: Optional[float] = None,
                  ) -> np.ndarray:

    loc = np.mean(arr) if loc is None else loc
    scale = np.std(arr) if scale is None else scale
    noise = np.random.normal(loc=np.mean(arr), scale=scale, size=arr.shape)
    return arr + noise


def sgn(val: np.ndarray) -> np.ndarray:

    if np.isscalar(val):
        return -1 if val < 0 else 1
    val = val if isinstance(val, np.ndarray) else np.array(val)
    out = np.ones_like(val)
    out[val < 0]  = -1
    return out


def squash(X: np.ndarray, beta: Number = 1.0) -> np.ndarray:
    return np.tanh(beta * X)


def zero_center(X, axis):
    return X - np.expand_dims(np.mean(X, axis=axis), axis)
