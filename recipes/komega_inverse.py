import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(1, os.path.join(current_path, "../src"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import copy
import argparse

from komega import KOmegaEquation
from utils import load_data, load_solution_komega
from objectives import BayesianObjective
from inverse import InverseSolver

class BayesianObjectiveU(BayesianObjective):
    def objective(self, val, param):
        J_obs = sum((val[::3] - self.val_target[::3])**2)/self.sigma_obs**2
        J_prior = sum((param - self.param_prior)**2)/self.sigma_prior**2
        J = 0.5*(J_obs + J_prior)
        return J

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--Retau", type=float, default=550.0, required=True, help="Reynolds number.")
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
    
    dirname ="base_solution"
    y, u, k, omega = load_solution_komega(dirname)
    eqn = KOmegaEquation(y, u, k, omega, Retau)
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
    qtarget[::3] = f(eqn.y)

    
    if restart:
        eqn.beta = np.loadtxt("beta").astype(np.complex)
    else:
        eqn.beta[:] = 1.0
        
    beta_prior = np.ones_like(eqn.beta)
    sigma_obs = 1e-10
    sigma_prior = 1.0

    
    eqn.objective = BayesianObjectiveU(qtarget, beta_prior, sigma_obs, sigma_prior)
    
    inverse_solver = InverseSolver(eqn)
    inverse_solver.maxiter = maxiter
    inverse_solver.stepsize = maxstep
    inverse_solver.algo = "bfgs"
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
    plt.savefig("figs/inverse_u.pdf")
    
    plt.figure(4)
    beta = get_beta(eqn.beta)
    plt.semilogx(eqn.yp, beta, 'r-')
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$\beta$")
    plt.legend(loc=2)
    plt.savefig("figs/inverse_%s_beta.pdf")
    
    
    
