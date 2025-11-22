import numpy as np


def calculate_energy(theta, theta_dot, alpha, alpha_dot, params):
    m_alpha = params["m_alpha"]
    m_theta = params["m_theta"]
    L = params["L"]
    r = params["r"]
    g = params["g"]

    E_kin_arm = (1 / 6) * m_theta * r**2 * theta_dot**2
    # Kinetic energy of pendulum (from eq. 9)
    E_kin_pend = (
        (1 / 2)
        * m_alpha
        * (
            (L**2 / 3) * (alpha_dot**2 + theta_dot**2 * np.sin(alpha) ** 2)
            + L * alpha_dot * theta_dot * r * np.cos(alpha)
            + r**2 * theta_dot**2
        )
    )

    # Potential energy of pendulum (from eq. 7)
    E_pot = (1 / 2) * L * m_alpha * g * (1 - np.cos(alpha))

    return E_kin_arm + E_kin_pend + E_pot


def analyze_energy_conservation(t, solutions, params):
    energies = []

    for i in range(len(t)):
        theta, theta_dot, alpha, alpha_dot = solutions[i]
        energy = calculate_energy(theta, theta_dot, alpha, alpha_dot, params)
        energies.append(energy)
    energies = np.array(energies)
    E_initial = energies[0]

    relative_error = (energies - E_initial) / E_initial * 100
    max_drift = np.max(np.abs(relative_error))
    final_drift = relative_error[-1]
    drift_rate = final_drift / t[-1]

    return {
        "energies": energies,
        "relative_error": relative_error,
        "max_drift": max_drift,
        "final_drift": final_drift,
        "drift_rate": drift_rate,
    }
