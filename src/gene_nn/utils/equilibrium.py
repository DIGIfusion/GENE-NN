"""
Utilities for reading EQDSK/ITERDB files and extracting local surrogate-model inputs.
"""

import re
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from gene_nn.utils.general_physics import calc_rho_toroidal_norm_from_q_and_psi, gene_coll, e, mu0


def _x0_list(x0: Union[float, int, Iterable[float]]) -> np.ndarray:
    if isinstance(x0, (float, int, np.floating, np.integer)):
        return np.array([float(x0)], dtype=float)
    arr = np.array(list(x0), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("x0 must be a float or a non-empty 1D iterable.")
    return arr

def _strictly_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(x)
    xs = np.asarray(x, dtype=float)[idx]
    ys = np.asarray(y, dtype=float)[idx]
    keep = np.concatenate(([True], np.diff(xs) > 0))
    return xs[keep], ys[keep]


def _a_over_L(profile: np.ndarray, rho: np.ndarray, x0: float) -> float:
    prof = np.asarray(profile, dtype=float)
    if np.any(prof <= 0):
        raise ValueError("Profile has non-positive values; cannot take log.")
    spl = CubicSpline(rho, np.log(prof))
    return -float(spl.derivative()(x0))


def read_EFIT_file(efit_file_name: str):
    with open(efit_file_name, "r") as file:
        header = file.readline().strip()

        try:
            nw, nh = map(int, header.split()[-2:])
        except Exception:
            raise ValueError(f"Could not read nw, nh from header: {header}")

        def read_scalar_lines(n_lines: int = 4):
            vals = []
            for _ in range(n_lines):
                line = file.readline().strip()
                found = re.findall(r"[-+]?\d*\.\d+(?:[eE][+-]?\d+)?", line)
                vals.extend(map(float, found))
            return vals

        def read_array(n: int):
            vals = []
            while len(vals) < n:
                line = file.readline().strip()
                found = re.findall(r"[-+]?\d*\.\d+(?:[eE][+-]?\d+)?", line)
                vals.extend(map(float, found))
            return np.asarray(vals[:n], dtype=float)

        scalars = read_scalar_lines(4)

        if len(scalars) < 13:
            raise ValueError(f"Expected at least 13 scalar values, got {len(scalars)}")

        rdim   = scalars[0]
        zdim   = scalars[1]
        rcentr = scalars[2]
        rleft  = scalars[3]
        zmid   = scalars[4]

        rmag   = scalars[5]
        zmag   = scalars[6]
        psiax  = scalars[7]
        psisep = scalars[8]
        bcentr = scalars[9]
        current = scalars[10]

        F = read_array(nw)
        p = read_array(nw)
        ffprime = read_array(nw)
        pprime = read_array(nw)
        psirz = read_array(nw * nh).reshape(nh, nw)
        qpsi = read_array(nw)

        counts_line = file.readline().strip().split()
        if len(counts_line) < 2:
            raise ValueError("Could not read boundary/limiter counts line from EQDSK")

        nbbbs = int(counts_line[0])
        limitr = int(counts_line[1])

        rbzb_data = read_array(nbbbs * 2)
        rlimzlim_data = read_array(limitr * 2)

        rb = rbzb_data[0::2]
        zb = rbzb_data[1::2]
        rlim = rlimzlim_data[0::2]
        zlim = rlimzlim_data[1::2]

    Rgrid = np.arange(nw, dtype=float) / float(nw - 1) * rdim + rleft
    Zgrid = np.arange(nh, dtype=float) / float(nh - 1) * zdim + (zmid - zdim / 2.0)

    psip_n = np.linspace(0.0, 1.0, nw)

    R_geom = 0.5 * (rb.max() + rb.min())
    a_geom = 0.5 * (rb.max() - rb.min())

    return (
        psip_n,
        Rgrid,
        Zgrid,
        F,
        p,
        ffprime,
        pprime,
        psirz,
        qpsi,
        rmag,
        zmag,
        nw,
        psiax,
        psisep,
        rb,
        zb,
        R_geom,
        a_geom,
    )

def extract_scalars_from_eqdsk_iterdb(
    eqdsk_path,
    iterdb_path,
    rho0,
):
    """
    Extract local input quantities from an EQDSK/ITERDB pair.

    The function reads equilibrium quantities from the EQDSK file and profile
    quantities from the ITERDB file, evaluates them at the requested radial
    locations, and returns one dictionary per x0 location. The returned
    quantities are intended for surrogate model inference.

    Required ITERDB profiles are rhot, ne, te, and vrot. If ti is missing, te is
    used as a fallback.
    """
    rho_vals = _x0_list(rho0)

    (
        psip_n,
        Rgrid,
        Zgrid,
        F,
        p,
        _ffprime,
        _pprime,
        psirz,
        qpsi,
        rmag,
        _zmag,
        _nw,
        psiax,
        psisep,
        _rb,
        _zb,
        _R_geom,
        a_geom,
    ) = read_EFIT_file(str(eqdsk_path))

    iterdb = read_iterdb(str(iterdb_path))

    required_iterdb_keys = ["rhot", "ne", "te", "vrot"]
    missing = [key for key in required_iterdb_keys if key not in iterdb]

    if missing:
        raise KeyError(
            f"ITERDB file is missing required profiles: {missing}. "
            f"Available profiles: {list(iterdb.keys())}"
        )

    rhot_prof = np.asarray(iterdb["rhot"], dtype=float)
    ne = np.asarray(iterdb["ne"], dtype=float)
    Te = np.asarray(iterdb["te"], dtype=float)
    vrot = np.asarray(iterdb["vrot"], dtype=float)

    if "ti" in iterdb:
        Ti = np.asarray(iterdb["ti"], dtype=float)
    else:
        print(
            "[warning] ITERDB file does not contain 'ti'. "
            "Using Te as Ti for Ti-dependent quantities."
        )
        Ti = Te.copy()

    psip_n = np.asarray(psip_n, dtype=float)
    qpsi = np.asarray(qpsi, dtype=float)
    F = np.asarray(F, dtype=float)
    p = np.asarray(p, dtype=float)

    rhot_eqdsk = calc_rho_toroidal_norm_from_q_and_psi(psip_n, qpsi)

    rhot_eq, qpsi_s = _strictly_increasing(rhot_eqdsk, qpsi)
    rhot_pr, ne_s = _strictly_increasing(rhot_prof, ne)
    rhot_te, Te_s = _strictly_increasing(rhot_prof, Te)
    rhot_ti, Ti_s = _strictly_increasing(rhot_prof, Ti)
    rhot_vrot, vrot_s = _strictly_increasing(rhot_prof, vrot)

    rhot_p_mhd, p_mhd_s = _strictly_increasing(rhot_eqdsk, p)

    q_spl = CubicSpline(rhot_eq, qpsi_s)
    p_mhd_spl = PchipInterpolator(rhot_p_mhd, p_mhd_s, extrapolate=False)

    ne_of_rho = PchipInterpolator(rhot_pr, ne_s, extrapolate=False)
    Te_of_rho = PchipInterpolator(rhot_te, Te_s, extrapolate=False)
    Ti_of_rho = PchipInterpolator(rhot_ti, Ti_s, extrapolate=False)
    vrot_of_rho = PchipInterpolator(rhot_vrot, vrot_s, extrapolate=False)

    F_axis = float(PchipInterpolator(psip_n, F, extrapolate=True)(0.0))
    R_axis = float(rmag)

    Bref = abs(F_axis / R_axis)

    a = float(a_geom)
    eps = float(a / R_axis)

    psiN_rz = (
        np.asarray(psirz, dtype=float) - float(psiax)
    ) / (
        float(psisep) - float(psiax)
    )

    F_of_psiN = PchipInterpolator(psip_n, F, extrapolate=False)

    R2D, _Z2D = np.meshgrid(
        np.asarray(Rgrid),
        np.asarray(Zgrid),
        indexing="xy",
    )

    F_rz = F_of_psiN(psiN_rz)
    Bphi_rz = F_rz / R2D

    inside = (psiN_rz <= 1.0) & np.isfinite(Bphi_rz)

    dR = float(np.mean(np.diff(Rgrid)))
    dZ = float(np.mean(np.diff(Zgrid)))
    dA = dR * dZ

    phiedge = float(Bphi_rz[inside].sum() * dA / (2.0 * np.pi))
    Lref = float(np.sqrt(2.0 * abs(phiedge) / Bref))

    rows = []

    rho_min = float(rhot_prof.min())
    rho_max = float(rhot_prof.max())

    for x0 in rho_vals:
        x0 = float(x0)

        if x0 < rho_min or x0 > rho_max:
            raise ValueError(f"x0={x0} outside ITERDB rho range [{rho_min}, {rho_max}]")

        q0 = float(q_spl(x0))
        dq_drho = float(q_spl.derivative()(x0))
        shat = float((x0 / q0) * dq_drho)

        omn_e = float(_a_over_L(ne_s, rhot_pr, x0))
        omt_e = float(_a_over_L(Te_s, rhot_te, x0))
        omt_i = float(_a_over_L(Ti_s, rhot_ti, x0))

        ne0 = float(ne_of_rho(x0))
        Te0 = float(Te_of_rho(x0))
        Ti0 = float(Ti_of_rho(x0))
        omegatorref = float(vrot_of_rho(x0))

        pe0 = ne0 * (Te0 * e)
        beta = float(2.0 * mu0 * pe0 / (Bref ** 2))

        dp_mhd_drho0 = float(p_mhd_spl.derivative()(x0))
        dpdx_pm = float(-(2.0 * mu0 / (Bref ** 2)) * dp_mhd_drho0)

        coll = float(gene_coll(ne0, Te0, Lref))

        rows.append(
            {
                "x0": x0,
                "q0": q0,
                "shat": shat,
                "omn_e": omn_e,
                "omt_e": omt_e,
                "omt_i": omt_i,
                "Bref": Bref,
                "Lref": Lref,
                "coll": coll,
                "beta": beta,
                "dpdx_pm": dpdx_pm,
                "omegatorref": omegatorref,
                "eps": eps,
            }
        )

    return rows


def read_iterdb(file_path: str):
    sec_start = re.compile(r"(?m)^.*UFILES ASCII FILE SYSTEM.*$")
    dep_label = re.compile(r"(?m)^\s*([A-Za-z0-9_+\-*/]+)\s*(\S+)?\s*;-\s*DEPENDENT VARIABLE LABEL-")
    nx_line = re.compile(r"(?m)^\s*(\d+)\s*;-\s*# OF X PTS-")
    ny_line = re.compile(r"(?m)^\s*(\d+)\s*;-\s*# OF Y PTS-")
    data_fol = re.compile(r"X,Y,F\(X,Y\)\s*DATA\s*FOLLOW", re.IGNORECASE)

    scinum = re.compile(
        r"""
        [+\-]?
        (?:\d+(?:\.\d*)?|\.\d+)
        (?:[EeDd][+\-]?\d+)?
        """,
        re.VERBOSE,
    )

    def extract_floats(text: str) -> np.ndarray:
        text = text.replace("D", "E").replace("d", "e").replace(",", " ")
        arr = np.fromstring(text, sep=" ")
        if arr.size > 0:
            return arr
        matches = scinum.findall(text)
        return np.array([float(x) for x in matches], dtype=float)

    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    starts = [m.start() for m in sec_start.finditer(text)] + [len(text)]

    rhotor = None
    variables = {}
    skipped = []

    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]

        m_dep = dep_label.search(block)
        m_nx = nx_line.search(block)
        m_ny = ny_line.search(block)
        m_df = data_fol.search(block)

        if not (m_dep and m_nx and m_ny and m_df):
            continue

        key = m_dep.group(1).strip().lower()
        nx = int(m_nx.group(1))
        ny = int(m_ny.group(1))

        data = extract_floats(block[m_df.end():])

        need = nx + ny + nx * max(ny, 1)
        if data.size < need:
            skipped.append(f"{key} (have {data.size}, need {need})")
            continue

        x = data[:nx]
        f = data[nx + ny:nx + ny + nx * max(ny, 1)]

        values = f if ny <= 1 else f.reshape(ny, nx)

        if rhotor is None:
            rhotor = x
        else:
            if rhotor.shape != x.shape or np.max(np.abs(rhotor - x)) > 1e-10:
                skipped.append(f"{key} (rhotor mismatch)")
                continue

        variables[key] = values

    if rhotor is None:
        raise ValueError("No valid sections parsed from ITERDB.")

    out = {"rhot": rhotor}
    for key in ["ne", "te", "ti", "ni", "vrot", "zeff"]:
        if key in variables:
            out[key] = variables[key]

    if skipped:
        print("[warning] Skipped ITERDB sections:", "; ".join(skipped))

    return out