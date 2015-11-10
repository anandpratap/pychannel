import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np

from stressomega import StressOmegaEquation
from utils import load_data, load_solution_stressomega

dirname ="base_solution"
y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
Retau = 550.0
eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
eqn.writedir = "solution"
eqn.dt = 1e2
eqn.force_boundary = False
eqn.tol = 1e-7
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

plt.figure(2)
plt.subplot(221)
plt.plot(eqn.yp, eqn.R11, 'r-', label=r'$k-\omega$')
plt.subplot(222)
plt.plot(eqn.yp, eqn.R12, 'r-', label=r'$k-\omega$')
plt.subplot(223)
plt.plot(eqn.yp, eqn.R22, 'r-', label=r'$k-\omega$')
plt.subplot(224)
plt.plot(eqn.yp, eqn.R33, 'r-', label=r'$k-\omega$')
plt.show()
