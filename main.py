import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from euler_backward import euler_backward
from euler_forward import euler_forward
from system_ode import furuta_pendulum


def main():
    params = {
        "m_alpha": 0.1,  # pendulum mass [kg]
        "m_theta": 0.2,  # arm mass [kg]
        "L": 0.3,  # pendulum length [m]
        "r": 0.15,  # arm length [m]
        "g": 9.81,  # gravity [m/s²]
        "tau": 0.0,  # input torque [N·m]
    }
    x0 = [0, 0, np.pi - 0.1, 0]

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)

    # Solve using RK45
    sol = solve_ivp(
        fun=lambda t, x: furuta_pendulum(t, x, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
    )

    # Solve using Euler forward
    h = 0.01  # step size
    t_euler_fwd, y_euler_fwd = euler_forward(
        f=lambda t, x: furuta_pendulum(t, x, params), t_span=t_span, y0=x0, h=h
    )

    # Solve using Euler backward
    t_euler_bwd, y_euler_bwd = euler_backward(
        f=lambda t, x: furuta_pendulum(t, x, params), t_span=t_span, y0=x0, h=h
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(sol.t, sol.y[0], label="RK45", linewidth=2)
    axes[0, 0].plot(
        t_euler_fwd, y_euler_fwd[:, 0], "--", alpha=0.7, label="Euler Forward"
    )
    axes[0, 0].plot(
        t_euler_bwd, y_euler_bwd[:, 0], ":", alpha=0.7, label="Euler Backward"
    )
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("θ [rad]")
    axes[0, 0].set_title("Arm angle")
    axes[0, 0].legend()

    axes[0, 1].plot(sol.t, sol.y[1], label="RK45", linewidth=2)
    axes[0, 1].plot(
        t_euler_fwd, y_euler_fwd[:, 1], "--", alpha=0.7, label="Euler Forward"
    )
    axes[0, 1].plot(
        t_euler_bwd, y_euler_bwd[:, 1], ":", alpha=0.7, label="Euler Backward"
    )
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("θ̇ [rad/s]")
    axes[0, 1].set_title("Arm angular velocity")
    axes[0, 1].legend()

    axes[1, 0].plot(sol.t, sol.y[2], label="RK45", linewidth=2)
    axes[1, 0].plot(
        t_euler_fwd, y_euler_fwd[:, 2], "--", alpha=0.7, label="Euler Forward"
    )
    axes[1, 0].plot(
        t_euler_bwd, y_euler_bwd[:, 2], ":", alpha=0.7, label="Euler Backward"
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("α [rad]")
    axes[1, 0].set_title("Pendulum angle")
    axes[1, 0].legend()

    axes[1, 1].plot(sol.t, sol.y[3], label="RK45", linewidth=2)
    axes[1, 1].plot(
        t_euler_fwd, y_euler_fwd[:, 3], "--", alpha=0.7, label="Euler Forward"
    )
    axes[1, 1].plot(
        t_euler_bwd, y_euler_bwd[:, 3], ":", alpha=0.7, label="Euler Backward"
    )
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("α̇ [rad/s]")
    axes[1, 1].set_title("Pendulum angular velocity")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
