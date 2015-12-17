import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(1, os.path.join(current_path, "../src"))
sys.path.insert(1, os.path.join(current_path, "../../pyheat/src"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import copy
import argparse
import pymc3 as mc

import theano
import theano.tensor as T 
from theano.compile.ops import as_op
from komega import KOmegaEquation
from utils import load_data, load_solution_komega
from objectives import BayesianObjective
from inverse import InverseSolver
from mcmc import MCMCSampler
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
    parser.add_argument("--nsamples", type=int, default=100, required=True, help="nsamples")
    parser.add_argument("--run", action="store_true", help="Run.")

    args = parser.parse_args()
    Retau = args.Retau
    dt = args.dt
    nsamples = args.nsamples
    tol = args.tol
  

    dirname ="bs"
    __base_y, __base_u, __base_k, __base_omega = load_solution_komega(dirname)
    
    @as_op(itypes=[theano.tensor.dvector], otypes=[theano.tensor.dvector])
    def function(beta):
        eqn = KOmegaEquation(__base_y, __base_u, __base_k, __base_omega, Retau)
        eqn.writedir = "solution"
        # first dt can be large as we restart from a good solution
        eqn.dt = 1e1
        eqn.force_boundary = False
        eqn.tol = tol
        eqn.beta[:] = beta.astype(np.complex)
        status = eqn.solve()
        if not status:
            tmp = eqn.q[::3].astype(np.float64)
        else:
            tmp = np.ones_like(eqn.y).astype(np.float64)*1e10
            print "NOT CONVERGED"
        return tmp


    n = np.size(__base_y)
    dns, wilcox, wilcox_kw = load_data()
    utau = Retau*1e-4*2.0
    ydns = dns.y*0.5
    ub = dns.u*utau
    f = interp1d(ydns, ub)
    
    data = f(__base_y).astype(np.float64)
    beta_prior = np.ones_like(__base_y).astype(np.float64)
    beta_map = np.loadtxt("beta").astype(np.float64)


    sigma_obs = 1e-10
    sigma_prior = 1.0
    tau_obs = np.eye(n)/sigma_obs**2
    tau_prior = np.eye(n)/sigma_prior**2

    sampler = MCMCSampler(function, data, tau_obs, beta_prior, tau_prior, beta_map, save="trace.sqlite", is_cov=False)
    with sampler.model:
        try:
            trace = mc.backends.sqlite.load("trace.sqlite")
            start = trace[-1]
        except:
            start = None
    if args.run:
        trace = sampler.sample(nsamples, start=start)
    else:
        pass
    mc.traceplot(trace)
    mc.summary(trace)
    
    beta_ = trace['beta'][:,:]
    beta_mean = np.mean(beta_, axis=0)
    beta_std = np.std(beta_, axis=0)
    plt.figure()
    plt.plot(beta_mean)
    plt.plot(trace["beta"][-1], '.')
    plt.plot(beta_map)
    plt.figure()
    plt.plot(function(beta_map[:]))
    plt.plot(data, 'x')
    plt.figure()
    plt.plot(beta_std)
    plt.show()


    
