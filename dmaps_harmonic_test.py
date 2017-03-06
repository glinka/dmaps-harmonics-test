"""A method to determine which DMAPS eigenvectors contain new directions/information

.. moduleauthor:: Alexander Holiday <holiday@alexanderholiday.com>

"""

import numpy as np

def compute_residuals(V, eps_med_scale):
    """Computes the local linear regression error for each of the DMAPS eigenvectors as a function of the previous eigenvectors. The linear regression kernel is a Gaussian with width median(distances)/eps_med_scale. In the returned residuals, res

    Args:
        V (array): DMAPS eigenvectors, stored in columns and ordered by decreasing eigenvalue. 
        eps_med_scale (float): the scale to use in the local linear regression kernel, typically around 3

    .. note:: V[:,0] is assumed to be the trivial constant eigenvector

    Returns:
        residuals (array): the residuals of each of the fitted functions. residuals[i] is close to 1 if V[:,i] parameterizes a new direction in the data.

    ..note:: residuals[0] should be ignored, and residuals[1] is always 1
    """

    neigvects = V.shape[1]

    residuals = np.zeros(neigvects)
    residuals[1] = 1.0

    for i in range(2,neigvects):
        residuals[i] = _local_linear_regression(V[:,i], V[:, 1:i], eps_med_scale)

    return residuals

def _local_linear_regression(y, X, eps_med_scale):
    """There is some math here"""

    n = X.shape[0]
    nvects = X.shape[1]

    K = np.empty((n, n))
    for i in range(n):
        K[i,i] = 0.0
        for j in range(i+1, n):
            K[i,j] = np.linalg.norm(X[i] - X[j])
            K[j,i] = K[i,j]
    
    eps = np.median(K)/eps_med_scale
    W = np.exp(-np.power(K/eps, 2))

    L = np.zeros((n,n));
    for i in range(n):
        Xx = np.hstack((np.ones((n,1)), X - np.ones((n, nvects))*X[i]))
        Xx2 = Xx.T*W[i]
        A = np.linalg.lstsq(np.dot(Xx2, Xx), Xx2)[0]
        L[i] = A[0]

    fx = np.dot(L, y)
    return np.sqrt(np.average(np.power((y-fx)/(1-np.diagonal(L)), 2)))/np.std(y, ddof=1)
