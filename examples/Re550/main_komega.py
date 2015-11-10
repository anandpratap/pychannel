import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np

from komega import KOmegaEquation
from utils import load_data, load_solution_komega

dirname ="base_solution"
y, u, k, omega = load_solution_komega(dirname)
Retau = 550.0
eqn = KOmegaEquation(y, u, k, omega, Retau)
eqn.writedir = "solution"
eqn.solve()
dns, wilcox = load_data()

plt.figure(1)
plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\omega$')
plt.semilogx(dns.yp[::5], dns.u[::5], 'bo', label=r'DNS', mfc="white")
plt.semilogx(wilcox.y, wilcox.u, 'g-', label=r'Wilcox $k-\omega$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.figure(2)
plt.semilogx(eqn.yp, eqn.kp, 'r-', label=r'$k-\omega$')
plt.semilogx(dns.yp[::5], dns.k[::5], 'bo', label=r'DNS', mfc="white")
plt.semilogx(wilcox.y, wilcox.k, 'g-', label=r'Wilcox $k-\omega$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.show()
