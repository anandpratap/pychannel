import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(1, os.path.join(current_path, "../src"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import copy
import argparse

from stressomega import StressOmegaEquation, get_beta
from utils import load_data, load_solution_stressomega
from objectives import BayesianObjective
from inverse import InverseSolver
__objective__choice = ["ALL", "U"]

class BayesianObjectiveU(BayesianObjective):
    def objective(self, val, param):
        J_obs = sum((val[::6] - self.val_target[::6])**2)/self.sigma_obs**2
        J_prior = sum((param - self.param_prior)**2)/self.sigma_prior**2
        J = 0.5*(J_obs + J_prior)
        return J

class BayesianObjectiveAll(BayesianObjective):
    def objective(self, val, param):
        weight_u = 1/np.mean(self.val_target[::6])*10.0
        weight_R11 = 1/np.mean(self.val_target[1::6])
        weight_R12 = 1/np.mean(self.val_target[2::6])
        weight_R22 = 1/np.mean(self.val_target[3::6])
        weight_R33 = 1/np.mean(self.val_target[4::6])

        J_obs = sum((val[::6] - self.val_target[::6])**2)/self.sigma_obs**2 * abs(weight_u)**2
        J_obs += sum((val[1::6] - self.val_target[1::6])**2)/self.sigma_obs**2 * abs(weight_R11)**2
        J_obs += sum((val[2::6] - self.val_target[2::6])**2)/self.sigma_obs**2 * abs(weight_R12)**2
        J_obs += sum((val[3::6] - self.val_target[3::6])**2)/self.sigma_obs**2 * abs(weight_R22)**2
        J_obs += sum((val[4::6] - self.val_target[4::6])**2)/self.sigma_obs**2 * abs(weight_R33)**2
        J_prior = sum((param - self.param_prior)**2)/self.sigma_prior**2
        J = 0.5*(J_obs + J_prior)
        return J

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--Retau", type=float, default=550.0, required=True, help="Reynolds number.")
    parser.add_argument("--objective", nargs=1, choices=__objective__choice, required=True, help="Type of objective function.")
    parser.add_argument("--dt", type=float, default=1.0, required=True, help="Solver time step.")
    parser.add_argument("--tol", type=float, default=1e-5, required=True, help="Solver tolerance.")
    parser.add_argument("--maxstep", type=float, default=0.05, required=True, help="Inverse max step size.")
    parser.add_argument("--maxiter", type=int, default=10, required=True, help="Inverse max number of iterations.")
    parser.add_argument("--restart", action="store_true", help="Read beta and restart from previous state.")
    args = parser.parse_args()

    Retau = args.Retau
    dt = args.dt
    maxstep = args.maxstep
    maxiter = args.maxiter
    restart = args.restart
    tol = args.tol
    objective = args.objective

    dirname ="base_solution"
    y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
    eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
    eqn.writedir = "solution"
    # first dt can be large as we restart from a good solution
    eqn.dt = 1e1
    eqn.force_boundary = False
    eqn.tol = tol
    eqn.solve()

    eqn.dt = dt
    eqn_prior = copy.deepcopy(eqn)
    up_prior = eqn.up.copy()

    qtarget = np.zeros_like(eqn.q)
    dns, wilcox, wilcox_kw = load_data()

    utau = Retau*eqn.nu*2.0
    ydns = dns.y*0.5
    ub = dns.u*utau
    R11b = -(dns.ub*utau)**2
    R22b = -(dns.vb*utau)**2
    R33b = -(dns.wb*utau)**2
    R12b = -dns.uv*utau*utau
    
    f = interp1d(ydns, ub)
    qtarget[::6] = f(eqn.y)
    f = interp1d(ydns, R11b)
    qtarget[1::6] = f(eqn.y)
    f = interp1d(ydns, R12b)
    qtarget[2::6] = f(eqn.y)
    f = interp1d(ydns, R22b)
    qtarget[3::6] = f(eqn.y)
    f = interp1d(ydns, R33b)
    qtarget[4::6] = f(eqn.y)
    
    if restart:
        eqn.beta = np.loadtxt("beta").astype(np.complex)
    else:
        eqn.beta[:] = 1.0
        
    beta_prior = np.ones_like(eqn.beta)
    sigma_obs = 1e-10
    sigma_prior = 1.0

    if objective[0] == "ALL":
        eqn.objective = BayesianObjectiveAll(qtarget, beta_prior, sigma_obs, sigma_prior)
    elif objective[0] == "U":
        eqn.objective = BayesianObjectiveU(qtarget, beta_prior, sigma_obs, sigma_prior)
    else:
        raise ValueError("Objective function not defined!")

    inverse_solver = InverseSolver(eqn)
    inverse_solver.maxiter = maxiter
    inverse_solver.stepsize = maxstep
    inverse_solver.algo = "sd"
    eqn = inverse_solver.solve()

    plt.ioff()
    plt.figure(11)
    plt.semilogx(eqn.yp, up_prior, 'g-', label=r'Prior')
    plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'Posterior')
    plt.semilogx(dns.yp[::5], dns.u[::5], 'b.', label=r'DNS')
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$u^+$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig("figs/inverse_%s_u.pdf"%objective)
    
    plt.figure(4)
    beta11, beta12, beta22, beta33 = get_beta(eqn.beta)
    plt.semilogx(eqn.yp, beta11, 'r-', label=r'R11')
    plt.semilogx(eqn.yp, beta12, 'g-', label=r'R12')
    plt.semilogx(eqn.yp, beta22, 'b-', label=r'R22')
    plt.semilogx(eqn.yp, beta33, 'c-', label=r'R33')
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$\beta$")
    plt.legend(loc=2)
    plt.savefig("figs/inverse_%s_beta.pdf"%objective)
    
    
    
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
    plt.savefig("figs/inverse_%s_reystress.pdf"%objective)
    plt.show()
