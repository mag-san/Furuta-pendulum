import math

import numpy as np

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


def f2():
    pass


def f3():
    pass


def f4():
    pass


# define the eiler forward method that will be used to solve the ODEs
def euler(v0, t0, h, f):
    v = [v0]
    t = [t0]
    while v[-1] <= t_max:
        v_new = v[-1] + h * f(v[-1], t[-1])
        t_new = t[-1] + h
        v.append(v_new)
        t.append(t_new)
    return t, v


# solving with various timesteps
t1, v1 = euler(0, 0, 0.3, f1)
t2, v2 = euler(0, 0, 0.3, f2)
t3, v3 = euler(0, 0, 0.3, f3)
t4, v4 = euler(0, 0, 0.3, f4)

# plot the solution
# TODO: add matplotlib and plot the good stuff
