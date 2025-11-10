import math

import matplotlib as plt
import numpy as np


def newton(f, f_der, n, x0):
    x = x0
    for i in range(n):
        x = x - f(x) / f_der(x)
    return x


# variables
# arm
theta = 0  # arm angle
m_theta = 2  # arm mass
r = 3  # arm length

# pendulum
alpha = 0  # angle
m_alpha = 3  # mass
L = 5  # length
I = L / 2  # half length

# other constants
g = 9.81  # gravitational acceleration
delta_h = (L / 2) * (
    1 - math.cos(alpha)
)  # height difference center pendulum rel. to ref. pos.
t_max = 10  # maximum t value
tau_theta = 0  # torque from the motor


# discuss differentiation of alpha and theta
# define the 4 ODEs that shall be sovlved
def f1():
    pass


def f1_der():
    pass


def f2():
    pass


def f2_der():
    pass


def f3():
    pass


def f3_der():
    pass


def f4():
    pass


def f4_der():
    pass


# define the eiler forward method that will be used to solve the ODEs
def impl_euler(v0, t0, h, f, f_der, n_n):
    v = [v0]
    t = [t0]
    while v[-1] <= t_max:

        def F(u):
            return u - v[-1] - h * f(t[-1], u)

        def F_der(u):
            return 1 - h * f_der(t[-1], u)

        v_new = newton(F, F_der, n_n, v[-1])
        t_new = t[-1] + h
        v.append(v_new)
        t.append(t_new)
    return t, v


n_newton_iterations = 20
# solving with various timesteps
t1, v1 = impl_euler(0, 0, 0.3, f1, f1_der, n_newton_iterations)
t2, v2 = impl_euler(0, 0, 0.3, f2, f2_der, n_newton_iterations)
t3, v3 = impl_euler(0, 0, 0.3, f3, f3_der, n_newton_iterations)
t4, v4 = impl_euler(0, 0, 0.3, f4, f4_der, n_newton_iterations)

# plot the solution
#
