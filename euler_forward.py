import numpy as np


def euler_forward(f, t_span, y0, h):
    """
    Euler forward method for solving ODEs.

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
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y
