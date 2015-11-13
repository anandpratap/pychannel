import sys
sys.path.insert(1, "../")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from laminar import LaminarEquation
from utils import load_data, load_solution_laminar

dirname ="."
N = 100
y = np.linspace(0.0, 0.5, N).astype(np.complex)
h = 0.5/(N-1)
u = np.zeros_like(y)
beta = np.ones_like(y)

Retau = 550.0
eqn = LaminarEquation(y, u, Retau)
eqn.writedir = "."
eqn.beta[:] = beta[:]
eqn.solve()
dJdbeta = eqn.calc_sensitivity()
# complex step calculation
dJdbeta_fd = np.zeros_like(y)
J_base = eqn.objfunc(eqn.q, eqn.beta)
dbeta = 1e-4
for i in range(len(beta)):
    beta[i] = beta[i] + dbeta
    eqn = LaminarEquation(y, u, Retau)
    eqn.writedir = "solution"
    eqn.beta[:] = beta[:]
    eqn.solve()
    dJdbeta_fd[i] = (eqn.objfunc(eqn.q, eqn.beta) - J_base)/dbeta
    beta[i] = beta[i] - dbeta


dJdbeta_cs = np.zeros_like(y)
dbeta = 1e-30
for i in range(len(beta)):
    beta[i] = beta[i] + 1j*dbeta
    eqn = LaminarEquation(y, u, Retau)
    eqn.writedir = "solution"
    eqn.beta[:] = beta[:]
    eqn.solve()
    dJdbeta_cs[i] = np.imag(eqn.objfunc(eqn.q, eqn.beta))/dbeta
    beta[i] = beta[i] - 1j*dbeta

assert(np.linalg.norm(eqn.uap - eqn.up)/N < h**2)
plt.figure(1)
plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'Discrete')
plt.semilogx(eqn.yp[::5], eqn.uap[::5], 'bo', label=r'Analytic', mfc="white")
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.figure(2)
plt.semilogx(eqn.yp, dJdbeta, 'ro-', label=r'Adjoint')
plt.semilogx(eqn.yp, dJdbeta_fd, 'g-', label=r'Finite Diff')
plt.semilogx(eqn.yp, dJdbeta_cs, 'b--', label=r'Complex Step')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.show()
