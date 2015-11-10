import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np

from ktau import KTauEquation
from utils import load_data, load_solution_ktau

dirname ="base_solution"
y, u, k, tau = load_solution_ktau(dirname)
Retau = 550.0
eqn = KTauEquation(y, u, k, tau, Retau)
eqn.writedir = "solution"
eqn.dt = 1e2
eqn.force_boundary = False
eqn.solve()
dns, wilcox = load_data()

plt.figure(1)
plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\tau$')
plt.semilogx(dns.yp[::5], dns.u[::5], 'bo', label=r'DNS', mfc="white")
plt.semilogx(wilcox.y, wilcox.u, 'g-', label=r'Wilcox $k-\tau$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.figure(2)
plt.loglog(eqn.yp[1:], eqn.kp[1:], 'r-', label=r'$k-\tau$')
plt.loglog(dns.yp[1::5], dns.k[1::5], 'bo', label=r'DNS', mfc="white")
plt.loglog(wilcox.y[1:], wilcox.k[1:], 'g-', label=r'Wilcox $k-\tau$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$k^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.figure(3)
plt.loglog(eqn.yp, eqn.taup, 'r-', label=r'$k-\tau$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$\tau^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.show()
