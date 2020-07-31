import abc
import logging
from numbers import Number
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Union,
)
import numpy as np



__all__ = [
    "HopfieldNetwork",
    "Frame",
]



class HopfieldNetwork(abc.ABC):

    #: Number of neurons (i.e., length of the state vector).
    _N: int

    #: State vector.
    _state: np.ndarray

    #: Weight matrix.
    _weights: np.ndarray

    #: Synaptic input.
    _I_syn: np.ndarray

    #: Neuronal bias.
    _bias: Union[Number, np.ndarray]

    #: Current iteration.
    _iteration: Optional[int]

    #: Data stored per iteration.
    _frames: List["Frame"]

    #: Flags
    _running: bool = False

    #: Variables to save on each iteration.
    record: List[str] = []

    #: Optional function to call at the end of each iteration.
    update_callback: Optional[Callable] = None


    def __init__(self, N: int):
        self._N = N
        self.clear()


    @property
    def N(self) -> int:
        return self._N


    @property
    def state(self) -> np.ndarray:
        return self._state


    @state.setter
    def state(self, X: Union[Number, np.ndarray]) -> None:
        X = X * np.ones(self._N) if np.isscalar(X) else np.asarray(X)
        assert X.shape == (self._N,)
        self._state = X


    @property
    def weights(self) -> np.ndarray:
        return self._weights


    @weights.setter
    def weights(self, val: Union[Number, np.ndarray]) -> None:
        val = np.array(val)
        if val.ndim == 0:
            self._weights = val * np.ones([self._N, self._N])
        else:
            if val.shape != (self._N, self._N):
                raise ValueError("weight matrix is malformed")
            self._weights = val
        np.fill_diagonal(self._weights, 0)


    @property
    def bias(self) -> np.ndarray:
        return self._bias


    @bias.setter
    def bias(self, b: Union[Number, np.ndarray]) -> None:
        b = b * np.ones(self._N) if np.isscalar(b) else np.asarray(b)
        if b.shape != (self._N,):
            raise ValueError("bad shape for bias variable")
        self._bias = b


    @property
    def I_syn(self) -> np.ndarray:
        return self._I_syn


    @property
    def iteration(self) -> int:
        return self._iteration


    @property
    def frames(self) -> List["Frame"]:
        return self._frames


    def clear(self) -> None:

        self._iteration = 0
        self.state = 0
        self.weights = 0
        self.bias = 0
        self._I_syn = np.zeros(self._N)
        self._frames = []

        self._started = False


    def get_frame(self):
        frame = Frame()
        for attr in self.record:
            setattr(frame, attr, getattr(self, attr))
        return frame


    def store_patterns(self, P: Iterable[np.ndarray]) -> None:

        weights = np.zeros([self._N, self._N])
        P = np.asarray(P)
        P = [P] if P.ndim == 1 else P

        for p in P:
            weights += np.outer(p, p)

        weights /= self._N
        np.fill_diagonal(weights, 0)
        self._weights = weights



    def run(self, num: int = 1) -> None:

        # If first run, store initial state as a frame first.
        if not self._started:
            self._iteration = 0
            self._frames = [self.get_frame()]
            self._running = True
            if self.update_callback:
                self.update_callback(self)

        for i in range(num):
            self._iteration += 1
            logging.info(f"iteration {self._iteration}")
            self.update()
            self._frames.append(self.get_frame())
            if self.update_callback:
                self.update_callback(self)



    def update(self) -> None:
        """ Abstract method to be implemented in subclasses.
        """
        raise NotImplementedError




class Frame:

    def __init__(self,
                 iteration: Optional[int] = None,
                 state: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 I_syn: Optional[np.ndarray] = None,
                 ):

        self.iteration = iteration
        self.state = state
        self.weights = weights
        self.I_syn = I_syn