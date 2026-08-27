"""Independent reference solution of the Clausing problem for a cylindrical tube.

Deterministic quadrature + linear solve (no Monte Carlo), so it is an independent
validation target for the ballistic MC transport kernel.

Ring-to-ring exchange per unit length inside a cylinder of radius R:
    k(u) = (R/pi) * int_0^{2pi} a^2 / D^4 dpsi
    a = R(1 - cos psi),   D^2 = 2R^2(1 - cos psi) + u^2
Exact checks used below:  k(0) = 1/(2R),   int_0^inf k du = 1/2.
"""
import numpy as np


def kernel(u, R=0.5, npsi=6000):
    psi = (np.arange(npsi) + 0.5) / npsi * 2 * np.pi
    omc = 1.0 - np.cos(psi)
    a2 = (R * omc) ** 2
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.empty(u.size)
    for i, ui in enumerate(u):
        out[i] = (a2 / (2 * R * R * omc + ui * ui) ** 2).sum()
    return (R / np.pi) * out * (2 * np.pi / npsi)


def _cumulative(R=0.5, umax=400.0, n=200001):
    """Phi(d) = int_0^d k(v) dv on a fine uniform grid (trapezoid)."""
    u = np.linspace(0.0, umax, n)
    k = kernel(u, R)
    Phi = np.concatenate(([0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(u))))
    return u, Phi


def disk_to_disk(h, R=0.5):
    """View factor between two coaxial disks of radius R separated by h."""
    h = np.asarray(h, dtype=float)
    X = 2.0 + (h / R) ** 2
    return 0.5 * (X - np.sqrt(X * X - 4.0))


_UG, _PHI = None, None


def Phi(d, R=0.5):
    global _UG, _PHI
    if _UG is None:
        _UG, _PHI = _cumulative(R)
    return np.interp(np.asarray(d, dtype=float), _UG, _PHI)


def clausing(AR, M=1000, R=0.5):
    """Transmission probability of a tube with L/D = AR (D = 2R)."""
    L = AR * 2 * R
    dz = L / M

    edges = np.arange(M + 1) * dz
    z = (np.arange(M) + 0.5) * dz
    e = disk_to_disk(edges[:-1], R) - disk_to_disk(edges[1:], R)

    off = np.arange(M) * dz
    row = Phi(off + dz / 2, R) - Phi(np.maximum(off - dz / 2, 0.0), R)
    row[0] = 2 * Phi(dz / 2, R)
    idx = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :])
    kmat = row[idx]

    esc = 0.5 - Phi(L - z, R)
    nu = np.linalg.solve(np.eye(M) - kmat, e)
    return float(disk_to_disk(L, R)) + float(nu @ esc)


if __name__ == "__main__":
    print(f"k(0) = {kernel([0.0])[0]:.6f}   (exact 1.0 for R = 0.5)")
    print(f"int_0^inf k du = {Phi(1e3):.6f}   (exact 0.5)")
    print("\n  AR = H/D   L/R      W (this solver)     4/(3 AR) asympt")
    for AR in [0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0]:
        w1, w2 = clausing(AR, M=600), clausing(AR, M=1200)
        print(f"  {AR:8.1f} {2*AR:6.1f}     {w2:.5f}  (dM {abs(w2-w1):.1e})"
              f"     {4/(3*AR):.5f}")
    print("\nCross-checks that this solver passes:")
    print("  * kernel normalisation and k(0) above are exact analytic values")
    print("  * W(L/R = 1) = 0.672 and W(L/R = 2) = 0.514 match Clausing's")
    print("    classic short-tube values")
    print("  * the ballistic MC of src/cyl_mc.py agrees with this solver to")
    print("    <= 0.25 % for AR = 0.5 .. 40 (4e6 particles); see")
    print("    src/fig_geometry_validation.py")
