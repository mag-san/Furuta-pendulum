import numpy as np
from scipy.optimize import fsolve


def euler_backward(f, t_span, y0, h):
    """
    Euler backward (implicit) method for solving ODEs.

    Parameters:
    -----------
    f : callable
        Function f(t, y) that returns dy/dt
    t_span : tuple
        (t0, tf) start and end times
    y0 : array_like
        Initial conditions
    h : float
        Step size

    Returns:
    --------
    t : ndarray
        Time points
    y : ndarray
        Solution at each time point (shape: [n_steps, n_states])
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n_steps = len(t)
    n_states = len(y0)

    y = np.zeros((n_steps, n_states))
    y[0] = y0

    for i in range(n_steps - 1):

        def F(u):
            return u - y[i] - h * f(t[i + 1], u)

        y_guess = y[i]

        y[i + 1] = fsolve(F, y_guess)

    return t, y
