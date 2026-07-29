import numpy as np

mu0 = 4e-7 * np.pi
e = 1.602176634e-19


def calc_rho_toroidal_norm_from_q_and_psi(psi, q): 
    """
    Calculate the toroidal flux surface coordinate rho_toroidal from the poloidal flux surface coordinate psi and safety factor q.
    equation: rho_toroidal = sqrt(1/(2*pi) * integral(q * dpsi))
    # d phi toroidal/d psi_polodial = q
    # --> phi_toroidal = integral(q * dpsi)
    # rho_toroidal = sqrt(phi_toroidal / max(phi_toroidal)) 
    """
    q_central_diff = 0.5 * (q[:-1] + q[1:])  # Central difference for q
    dpsi = np.diff(psi)
    dphi_toroidal = q_central_diff * dpsi
    phit = np.cumsum(dphi_toroidal)
    phit = np.concatenate(([0], phit))  # Add zero for the first element
    rho_toroidal_norm = np.sqrt(phit / np.max(phit))
    return rho_toroidal_norm

def gene_coll(nref_m3: float, Tref_eV: float, Lref_m: float) -> float:
    lnLambda = 24.0 - np.log(np.sqrt(nref_m3 * 1e-6) / Tref_eV)
    n19 = nref_m3 / 1e19
    TkeV = Tref_eV / 1e3
    return float(2.3031e-5 * (Lref_m * n19 / (TkeV**2)) * lnLambda)