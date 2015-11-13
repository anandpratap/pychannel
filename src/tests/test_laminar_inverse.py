import sys
sys.path.insert(1, "../")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from laminar import LaminarEquation
from utils import load_data, load_solution_laminar
from objectives import BayesianObjective
from inverse import InverseSolver

dirname ="."
N = 30
y = np.linspace(0.0, 0.5, N).astype(np.complex)
h = 0.5/(N-1)
u = np.zeros_like(y)
beta = np.ones_like(y)

Retau = 550.0
eqn = LaminarEquation(y, u, Retau)
eqn.writedir = "."
eqn.beta[:] = beta[:]
eqn.solve()

sigma_prior = 1.0
sigma_obs = 1e-10
q_target = eqn.q[:] + np.random.randn(np.size(eqn.q))*sigma_obs

eqn.beta[:] = 2.0
eqn.solve()
q_prior = eqn.q.copy()
beta_prior = eqn.beta.copy()


eqn.objective = BayesianObjective(q_target, beta_prior, sigma_obs, sigma_prior)

inverse_solver = InverseSolver(eqn)
inverse_solver.maxiter = 100
eqn = inverse_solver.solve()

plt.figure(1)
plt.plot(y, q_prior, "g-", label="Prior")
plt.plot(y, eqn.q, "r-", label="Posterior")
plt.plot(y, q_target, "bx", label="Target")
plt.xlabel("y")
plt.ylabel(r"u")
plt.legend(loc=2)

plt.figure(2)
plt.plot(y, beta_prior, "g-", label="Prior")
plt.plot(y, eqn.beta, "r-", label="Posterior")
plt.plot(y, np.ones_like(eqn.beta), "bx", label="Targer")
plt.xlabel("y")
plt.ylabel(r"$\beta$")
plt.legend()
plt.show()

