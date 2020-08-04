import logging
from typing import Optional
import numpy as np
from hopfield_base import HopfieldNetwork
import math

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
    net = ContinuousHopfieldNetwork(N)
    n_patterns = 5
    ind_state_0 = 2
    whiten = True
    prune = 0.98

    # Load patterns, initialize transform, and train network.
    P = load_patterns(n_patterns)
    P = zero_center(P, 1)
    P0 = P.copy()
    tform = WhitenTransform("zca") if whiten else IdentityTransform()
    tform.fit(P)
    P = tform(P)
    net.store_patterns(P)

    weights = net.weights.flatten()
    K = len(weights)
    ixs = np.random.choice(K, round(K * prune))
    weights[ixs] = 0
    weights = weights.reshape(N, N)
    if whiten:
        weights *= N * 2
    net._weights = weights


    # Initialize state vector.
    state_0 = P0[ind_state_0]
    state_0_noisy = shuffle_degrade(state_0, 0.1)
    state_0_noisy = noise_degrade(state_0_noisy, 0.1)
    net.state = tform(state_0_noisy)

    # Setup plotting.
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(state_0.reshape(n_rows, n_cols), cmap="gray")
    ax1.set_title("template")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(state_0_noisy.reshape(n_rows, n_cols), cmap="gray")
    ax2.set_title("start")

    ax3 = fig.add_subplot(1, 3, 3)
    cdata = np.zeros([n_rows, n_cols, 4])
    cdata[:, :, -1] = 1
    cimage = ax3.imshow(cdata)
    ax3.set_xticks([])
    ax3.set_yticks([])


    def update_plot(net: HopfieldNetwork) -> None:

        state = tform.inv(net.state)
        state = (state - state.min()) / (state.max() - state.min())
        state = state.reshape(n_cols, n_rows)
        cdata[:, :, 0:3] = np.expand_dims(state, 2)
        cimage.set_data(cdata)
        ax3.set_title(f"iteration: {net.iteration}")
        plt.pause(0.2)


    net.update_callback = update_plot
    net.record = ["state", "I_syn"]
    net.run(25)
    

    
##============ OUR CODE

    
def calcPerf():  
    templateMemory = state_0.reshape(n_rows, n_cols)
    newTemplateMemory = templateMemory
    
    for x in range(n_rows):
        for y in range(n_cols):
            newTemplateMemory[x,y] = templateMemory[x,y] - (2/3)
    
    networkRecall = cdata[0:28, 0:28, 0]
    NNR = networkRecall
    
    for x in range(28):
        for y in range(28):
            a = networkRecall[x,y]
            NNR[x,y] = a*2 - 1
    
    difference = np.zeros((28,28))
    
    for x in range(28):
        for y in range(28):
            a = NNR[x,y]
            b = newTemplateMemory[x,y]
            difference[x,y] = abs(a-b)
            
    totalError = np.sum(difference)
    averageError = totalError/784

    print("Average Error = " + str(format(averageError, '0.2f')))