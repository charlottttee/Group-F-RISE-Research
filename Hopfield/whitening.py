from typing import Optional
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg





def get_whitening_matrix(
    X: np.ndarray,
    method: str = "pca",
    epsilon: float=0.1,
    axis: int = 1,
    ) -> np.ndarray:

    """
    Whitens the input matrix X using specified whitening method.

    Parameters:
    ----------

    X: np.ndarray
        Input data matrix with data examples along the first dimension

    method: str
        Whitening method. Must be one of 'zca', 'zca_cor', 'pca', 'pca_cor',
        or 'cholesky'.



    Parameters
    ----------

    Returns
    -------

    W : np.ndarray
        Whitening matrix. Use np.dot(data, W.T).

    """

    # Ensure data is 2D.
    if not X.ndim == 2:
        warnings.warn("data expected to be 2D. reshaping.")
    X = X.reshape((-1, np.prod(X.shape[1:])))

    # Zero-center along an axis.
    X = X - np.expand_dims(np.mean(X, axis=axis), axis)
    Sigma = np.dot(X.T, X) / X.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        fac = Lambda + epsilon
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(fac)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(fac)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / fac), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        fac = Theta + epsilon
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(fac)), G.T)),
                       np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(fac)), G.T),
                       np.linalg.inv(V_sqrt))
    else:
        raise ValueError('Whitening method f"{method}" not found.')

    return W


def whiten(
    X: np.ndarray,
    **kw,
    ) -> np.ndarray:

    # Ensure data is 2D.
    if not X.ndim == 2:
        warnings.warn("data expected to be 2D. reshaping.")
    X = X.reshape((-1, np.prod(X.shape[1:])))

    # Zero-center along an axis.
    axis = kw.get("axis", 1)
    X = X - np.expand_dims(np.mean(X, axis=axis), axis)
    # Get the whitening matrix, and whiten the data.
    W = get_whitening_matrix(X, **kw)
    return np.dot(X, W.T)



class Transform:


    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def inv(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityTransform(Transform):

    def fit(self, X: np.ndarray):
        return

    def inv(self, X: np.ndarray) -> np.ndarray:
        return X

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X


class WhitenTransform(Transform):


    _W: Optional[np.ndarray] = None
    _W_inv: Optional[np.ndarray] = None


    def __init__(self,
                 method: str = "pca",
                 axis: int = 1,
                 ):
        self._method = method
        self._axis = axis

    @property
    def method(self) -> str:
        return self._method

    @property
    def axis(self) -> str:
        return self._axis

    def fit(self, X: np.ndarray):
        self._W = get_whitening_matrix(X, self._method, self._axis)


    def inv(self, X: np.array) -> np.ndarray:
        if self._W_inv is None:
            self._W_inv = linalg.inv(self._W.T)
        return np.dot(X, self._W_inv)


    def __call__(self, X: np.array) -> np.ndarray:
        return np.dot(X, self._W.T)


if __name__ == "__main__":

    from mnistutils import load_mnist

    n_images = 10000
    method = "pca"
    axis = 1

    # Load data.
    mnist = load_mnist()
    X = mnist["train_images"][:n_images] / 255.0

    # Get whitening matrix and its inverse.
    W = get_whitening_matrix(X, method=method, axis=axis)
    W_inv = linalg.inv(W.T)

    ix = 9
    im = X[ix]
    im_centered = im - np.mean(im)
    im_whitened = np.dot(im_centered, W.T)
    im_dewhitened = np.dot(im_whitened, W_inv)
    im_diff = im_dewhitened - im_centered

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(im_centered.reshape(28, 28), cmap="gray")
    ax1.set_title('centered')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(im_whitened.reshape(28, 28), cmap="gray")
    ax2.set_title(f'whitened ({method})')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(im_dewhitened.reshape(28, 28), cmap="gray")
    ax3.set_title('de-whitened')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(im_diff.reshape(28, 28), cmap="gray")
    ax4.set_title('diff')

    plt.show()
