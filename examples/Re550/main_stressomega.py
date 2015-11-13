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


plt.figure(1)
plt.semilogx(eqn.yp, eqn.up, 'g-', label=r'$stress-\omega$')
plt.semilogx(dns.yp[::5], dns.u[::5], 'b.', label=r'DNS')
plt.semilogx(wilcox.y, wilcox.u, 'r--', label=r'Wilcox $stress-\omega$')
plt.semilogx(wilcox_kw.y, wilcox_kw.u, 'c--', label=r'Wilcox $k-\omega$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("figs/stress_omega_u.pdf")

plt.figure(2)
plt.loglog(eqn.yp, eqn.kp, 'g-', label=r'$stress-\omega$')
plt.loglog(dns.yp[::5], dns.k[::5], 'b.', label=r'DNS')
plt.loglog(wilcox.y, wilcox.k, 'r--', label=r'Wilcox $stress-\omega$')
plt.semilogx(wilcox_kw.y, wilcox_kw.k, 'c--', label=r'Wilcox $k-\omega$')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$k^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("figs/stress_omega_k.pdf")

plt.figure(3)
plt.subplot(222)
plt.plot(eqn.yp, -eqn.R11, 'g-', label=r'$stress-\omega$')
plt.plot(dns.yp[::5], dns.ub[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$uu^+$")
plt.gca().set_ylim(bottom=0)

plt.subplot(221)
plt.plot(eqn.yp, eqn.R12, 'g-', label=r'$stress-\omega$')
plt.plot(dns.yp[::5], -dns.uv[::5], 'b.', label=r'DNS')
plt.plot(wilcox.y, -wilcox.uv, 'r--', label=r'Wilcox $stress-\omega$')
plt.legend(bbox_to_anchor=(2.4, 1.1))
plt.xlabel(r"$y^+$")
plt.ylabel(r"$-uv^+$")
plt.gca().set_ylim(bottom=0)

plt.subplot(223)
plt.plot(eqn.yp, -eqn.R22, 'g-', label=r'$stress-\omega$')
plt.plot(dns.yp[::5], dns.vb[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$vv^+$")
plt.gca().set_ylim(bottom=0)

plt.subplot(224)
plt.plot(eqn.yp, -eqn.R33, 'g-', label=r'$stress-\omega$')
plt.plot(dns.yp[::5], dns.wb[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$ww^+$")
plt.gca().set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figs/stress_omega_reystress.pdf")



plt.figure(4)
dJdbeta11, dJdbeta12, dJdbeta22, dJdbeta33 = get_beta(dJdbeta)
plt.semilogx(eqn.yp, dJdbeta11, '-', label=r'R11')
plt.semilogx(eqn.yp, dJdbeta12, '--', label=r'R12')
plt.semilogx(eqn.yp, dJdbeta22, '-', label=r'R22')
plt.semilogx(eqn.yp, dJdbeta33, '-', label=r'R33')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$dJdbeta^+$")
plt.legend(loc=2)
plt.tight_layout()

plt.show()
