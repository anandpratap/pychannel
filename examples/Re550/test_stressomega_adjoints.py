import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np

from stressomega import StressOmegaEquation, get_beta
from utils import load_data, load_solution_stressomega

dirname ="base_solution"
y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
Retau = 550.0
eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
eqn.writedir = "solution/stress_omega"
eqn.dt = 1e2
eqn.force_boundary = False
eqn.tol = 1e-7
eqn.solve()
dJdbeta = eqn.calc_sensitivity()
dns, wilcox, wilcox_kw = load_data()
dJdbeta_cs = np.zeros_like(dJdbeta)

dbeta = 1e-30
for i in range(len(dJdbeta)):
    eqn.beta[i] = eqn.beta[i] + 1j*dbeta
    eqn.solve()
    dJdbeta_cs[i] = np.imag(eqn.objfunc(eqn.q, eqn.beta))/dbeta
    eqn.beta[i] = eqn.beta[i] - 1j*dbeta

plt.figure(4)
dJdbeta11, dJdbeta12, dJdbeta22, dJdbeta33 = get_beta(dJdbeta)
dJdbeta11_cs, dJdbeta12_cs, dJdbeta22_cs, dJdbeta33_cs = get_beta(dJdbeta_cs)
plt.semilogx(eqn.yp, dJdbeta11, 'r-', label=r'R11')
plt.semilogx(eqn.yp, dJdbeta12, 'g-', label=r'R12')
plt.semilogx(eqn.yp, dJdbeta22, 'b-', label=r'R22')
plt.semilogx(eqn.yp, dJdbeta33, 'c-', label=r'R33')
plt.semilogx(eqn.yp, dJdbeta11_cs, 'rx')
plt.semilogx(eqn.yp, dJdbeta12_cs, 'gx')
plt.semilogx(eqn.yp, dJdbeta22_cs, 'bx')
plt.semilogx(eqn.yp, dJdbeta33_cs, 'cx')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$dJdbeta^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("figs/stress_omega_adjoints.pdf")
plt.show()
