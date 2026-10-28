import sys
print([m for m in sys.modules if m.startswith("mathlibpkg")])


import numpy as np
from mathlibpkg.analysis.integral_1D import Integral1D

import matplotlib.pyplot as plt

def gauss(x):
    return np.exp(-x**2)

def fourier_trafo(func, k, N=2000, interval=(-10.0, 10.0)):
    """F(k) = ∫ func(x) e^{ikx} dx, numerisch auf [a, b]."""
    F = np.empty(len(k), dtype=complex)
    for j, kj in enumerate(k):
        # RK4 ruft F(t, y) auf: t = Integrationsvariable, y wird ignoriert
        integrand = lambda t, y: func(t) * np.exp(1j * kj * t)
        _, y = Integral1D.improper_integral(integrand, N, list(interval))
        F[j] = y[-1]          # Endwert der Stammfunktion = bestimmtes Integral
    return F

if __name__ == "__main__":
    k = np.linspace(-10, 10, 1000)
    F = fourier_trafo(lambda x: np.exp(x), k)
    plt.plot(k,F, label="Fourier")
    #F_exakt = np.sqrt(np.pi) * np.exp(-k**2 / 4)
    #plt.plot(k,F_exakt(k),label="exakt")
    #plt.legend()
    plt.show()
    #print("max. Fehler:", np.max(np.abs(F - F_exakt)))