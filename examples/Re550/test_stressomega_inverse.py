import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import copy

from stressomega import StressOmegaEquation, get_beta
from utils import load_data, load_solution_stressomega
from objectives import BayesianObjective
from inverse import InverseSolver

class BayesianObjectiveU(BayesianObjective):
    def objective(self, val, param):
        J_obs = sum((val[::6] - self.val_target)**2)/self.sigma_obs
        J_prior = sum((param - self.param_prior)**2)/self.sigma_prior**2
        J = 0.5*(J_obs + J_prior)
        return J


dirname ="base_solution"
y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
Retau = 5.467390699999999697e+02
eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
eqn.writedir = "solution/stress_omega"
eqn.dt = 1e2
eqn.force_boundary = False
eqn.tol = 1e-5
eqn.solve()
eqn_prior = copy.deepcopy(eqn)
up_prior = eqn.up.copy()

dns = np.loadtxt("data/DNSsol.dat")
utau = Retau*eqn.nu*2.0
ydns = dns[:,0]*0.5
udns = dns[:,2]*utau
f = interp1d(ydns, udns)
utarget = f(eqn.y)



beta_prior = eqn.beta.copy()
sigma_obs = 1e-10
sigma_prior = 1.0

eqn.objective = BayesianObjectiveU(utarget, beta_prior, sigma_obs, sigma_prior)
inverse_solver = InverseSolver(eqn)
inverse_solver.maxiter = 10
inverse_solver.stepsize = 0.1
inverse_solver.algo = "gn"
eqn = inverse_solver.solve()

plt.figure(11)
plt.semilogx(eqn.yp, up_prior, 'g-', label=r'Prior')
plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'Posterior')
plt.semilogx(eqn.yp[::5], utarget[::5]/utau, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("figs/inverse_matchu_u.pdf")

plt.figure(4)
beta11, beta12, beta22, beta33 = get_beta(eqn.beta)
plt.semilogx(eqn.yp, beta11, 'r-', label=r'R11')
plt.semilogx(eqn.yp, beta12, 'g-', label=r'R12')
plt.semilogx(eqn.yp, beta22, 'b-', label=r'R22')
plt.semilogx(eqn.yp, beta33, 'c-', label=r'R33')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$\beta$")
plt.legend()
plt.savefig("figs/inverse_matchu_beta.pdf")



dns, wilcox, wilcox_kw = load_data()
plt.figure(3)
plt.subplot(222)
plt.plot(eqn.yp, -eqn_prior.R11, 'g-', label=r'Prior')
plt.plot(eqn.yp, -eqn.R11, 'r-', label=r'Posterior')
plt.plot(dns.yp[::5], dns.ub[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$uu^+$")
plt.gca().set_ylim(bottom=0)


plt.subplot(221)
plt.plot(eqn.yp, eqn_prior.R12, 'g-', label=r'Prior')
plt.plot(eqn.yp, eqn.R12, 'r-', label=r'Posterior')
plt.plot(dns.yp[::5], -dns.uv[::5], 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$-uv^+$")
plt.gca().set_ylim(bottom=0)

plt.subplot(223)
plt.plot(eqn.yp, -eqn_prior.R22, 'g-', label=r'Prior')
plt.plot(eqn.yp, -eqn.R22, 'r-', label=r'Posterior')
plt.plot(dns.yp[::5], dns.vb[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$vv^+$")
plt.gca().set_ylim(bottom=0)

plt.subplot(224)
plt.plot(eqn.yp, -eqn_prior.R33, 'g-', label=r'Prior')
plt.plot(eqn.yp, -eqn.R33, 'r-', label=r'Posterior')
plt.plot(dns.yp[::5], dns.wb[::5]**2, 'b.', label=r'DNS')
plt.xlabel(r"$y^+$")
plt.ylabel(r"$ww^+$")
plt.gca().set_ylim(bottom=0)
plt.tight_layout()
plt.legend()
plt.savefig("figs/inverse_matchu_reystress.pdf")

plt.show()
