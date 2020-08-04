from typing import Optional
import numpy as np
from hopfield_base import HopfieldNetwork
from mnistutils import load_mnist
from utils import squash

__all__ = [
    "ContinuousHopfieldNetwork",
]


class ContinuousHopfieldNetwork(HopfieldNetwork):

    #: Time constant.
    tau = 0.1

    #: Decay rate.
    decay_rate = 0.1


    def update(self):
        state = self._state
        decay = -self.decay_rate * state
        self._I_syn = np.dot(self._weights, state)
        state = state + self.tau * (self._bias + decay + self._I_syn)
        state = squash(state)
        self._state = state



def load_patterns(n_patterns: Optional[int] = None) -> np.ndarray:
    """
    Returns training images from the mnist dataset in hopfield-network form.
    More specifically, 8-bit images from (0, 255) are converted to
    floating point images having values in {-1, 1}.
    """
    mnist = load_mnist()
    out = mnist["images"][:n_patterns]
    out = 2 * (out.astype(np.float32) / 255.0) - 1
    return out