import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(1, os.path.join(current_path, "../src"))
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import array, linalg, dot
import scipy.io as io
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

class BayesianObjectiveStats(BayesianObjective):
    def set_index(self, index):
        self.index = index
        
    def objective(self, val, param):
        weight_u = 1/np.mean(self.val_target[::6])*10.0
        weight_R11 = 1/np.mean(self.val_target[1::6])
        weight_R12 = 1/np.mean(self.val_target[2::6])
        weight_R22 = 1/np.mean(self.val_target[3::6])
        weight_R33 = 1/np.mean(self.val_target[4::6])
        n = np.size(self.val_target)/6
        J_obs = np.zeros(5*n, dtype=np.complex)
        J_obs[0:n] = (val[::6] - self.val_target[::6]) * abs(weight_u)
        J_obs[n:2*n] = (val[1::6] - self.val_target[1::6]) * abs(weight_R11)
        J_obs[2*n:3*n] = (val[2::6] - self.val_target[2::6]) * abs(weight_R12)
        J_obs[3*n:4*n] = (val[3::6] - self.val_target[3::6]) * abs(weight_R22)
        J_obs[4*n:5*n] = (val[4::6] - self.val_target[4::6]) * abs(weight_R33)
        J = J_obs[self.index]
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
    parser.add_argument("--nsamples", type=int, default=10, required=True, help="Number of samples.")
    args = parser.parse_args()

    Retau = args.Retau
    dt = args.dt
    maxstep = args.maxstep
    maxiter = args.maxiter
    restart = args.restart
    tol = args.tol
    objective = args.objective
    nsamples = args.nsamples

    sigma_obs = 1e-10
    sigma_prior = 0.5


    dirname ="base_solution"
    y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
    eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
    eqn.writedir = "/tmp"
    # first dt can be large as we restart from a good solution
    eqn.force_boundary = False
    eqn.tol = tol
    eqn.dt = 1e1
    eqn.solve()
    beta_prior = np.ones_like(eqn.beta)
    eqn_prior = copy.deepcopy(eqn)
    up_prior = eqn.up.copy()

    Q_prior = np.zeros([np.size(eqn.q), nsamples])

    sample = 0
    while sample < nsamples:
        print "\rSampling prior %i of %i"%(sample, nsamples),
        sys.stdout.flush()
        eqn.dt = dt
        eqn.q[:] = eqn_prior.q[:]
        eqn.beta[:] = beta_prior + np.random.randn(np.size(beta_prior))*sigma_prior
        print np.mean(eqn.beta)
        print np.std(eqn.beta)
        try:
            ierr = eqn.solve()
            if ierr == False:
                Q_prior[:, sample] = eqn.q[:]
                sample += 1
        except:
            pass
    
    dirname ="solution"
    y, u, R11, R12, R22, R33, omega = load_solution_stressomega(dirname)
    eqn = StressOmegaEquation(y, u, R11, R12, R22, R33, omega, Retau)
    eqn.beta[:] = np.loadtxt("inverse_solution/beta.%i"%maxiter)
    eqn.writedir = "/tmp"
    # first dt can be large as we restart from a good solution
    eqn.force_boundary = False
    eqn.dt = 1e1
    eqn.tol = tol
    eqn.solve()
    eqn.postprocess(eqn.q)
    eqn_map = copy.deepcopy(eqn)
    up_map = eqn.up.copy()

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
    
        
    beta_map = eqn.beta[:]
    eqn.objective = BayesianObjectiveStats(qtarget, beta_prior, sigma_obs, sigma_prior)

    n = np.size(qtarget)/6
    try:
        dat = io.loadmat("H.mat")
        H = data["H"]
        print 1000*"#"
        print "Using Hessian from file"
    except:
        njac = 10
        Jac = np.zeros([njac, 4*n])
        for i in range(njac):
            print "\rBuilding jac %i of %i"%(i, 5*n),
            sys.stdout.flush()
            eqn.objective.set_index(i)
            dJdbeta = eqn.calc_sensitivity()
            Jac[i,:] = dJdbeta[:]
        H = Jac.transpose().dot(Jac)/sigma_obs**2 + np.eye(5*n)/sigma_prior**2
        io.savemat("H.mat", {"H":H})

    Cov = linalg.inv(H)
    R = linalg.cholesky(Cov)
    Q_post = np.zeros([np.size(eqn.q), nsamples])

    sample = 0
    while sample < nsamples:
        print "\rSampling post %i of %i"%(sample, nsamples),
        sys.stdout.flush()
        eqn.dt = dt
        eqn.q[:] = eqn_map.q[:]
        eqn.beta[:] = beta_map + R.dot(np.random.randn(np.shape(beta_map)))
        try:
            ierr = eqn.solve()
            if ierr == False:
                Q_post[:, sample] = eqn.q[:]
                sample += 1
        except:
            pass
    io.savemat("stats.mat", {"H":H, "Cov":Cov, "beta_map":beta_map, "Q_prior":Q_prior, "Q_post":Q_post})
