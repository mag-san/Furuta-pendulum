import numpy as np


def runge_kutta_4(f, t_span, y0, h):
    """
    Fourth-order Runge-Kutta (RK4) method for solving ODEs.

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
        # Four stages of RK4
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(t[i] + h, y[i] + h * k3)

        # Weighted average
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y
