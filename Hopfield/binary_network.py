import logging
from typing import Optional
import numpy as np
from hopfield_base import HopfieldNetwork
from utils import sgn



__all__ = [
    "BinaryHopfieldNetwork",
]


class BinaryHopfieldNetwork(HopfieldNetwork):

    def update(self) -> None:

        # Compute synaptic input.
        self._I_syn = np.dot(self._weights, self._state)
        self._state = sgn(self._bias + self._I_syn)



def load_patterns(n_patterns: Optional[int] = None) -> np.ndarray:
    """
    Returns training images from the mnist dataset in hopfield-network form.
    More specifically, 8-bit images from (0, 255) are converted to
    floating point images having values in {-1, 1}.
    """
    mnist = load_mnist()
    out = mnist["train_images"][:n_patterns]
    out = 2 * (out.astype(np.float32) / 255.0) - 1
    out = sgn(out)
    return out



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mnistutils import load_mnist
    from utils import *
    from whitening import *


    # Setup logging and some constants.
    logging.basicConfig(level=logging.INFO)
    n_rows, n_cols = 28, 28
    N = n_rows * n_cols

    # Create the network, and specify parameters.
    net = BinaryHopfieldNetwork(N)
    n_patterns = 3
    ind_state_0 = 0


    # Load patterns, initialize transform, and train network.
    P = load_patterns(n_patterns)
    net.store_patterns(P)

    # Initialize state vector.
    state = P[ind_state_0]
    state = shuffle_degrade(state, 0.3)
    net.state = state

    # Setup plotting.
    fig, ax = plt.subplots()
    cdata = np.zeros([n_rows, n_cols, 4])
    cdata[:, :, -1] = 1
    cimage = ax.imshow(cdata)
    ax.set_xticks([])
    ax.set_yticks([])

    def update_plot(net: HopfieldNetwork) -> None:

        state = net.state
        state = state.reshape(n_cols, n_rows)
        cdata[state < 0] =  (0.0, 0.0, 0.0, 1.0)
        cdata[state >= 0] = (1.0, 1.0, 1.0, 1.0)
        cimage.set_data(cdata)
        ax.set_title(f"iteration: {net.iteration}")
        plt.pause(0.5)


    net.update_callback = update_plot
    net.record = ["state"]
    net.run(10)




