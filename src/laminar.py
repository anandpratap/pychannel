import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from utils import calc_dp, load_solution_laminar, load_data
from schemes import diff, diff2
from objectives import TestObjective

class LaminarEquation(object):
    def __init__(self, y, u, Retau):
        self.y = np.copy(y)
        self.q = np.copy(u.astype(np.complex))
        self.Retau = Retau

        #
        self.n = np.size(y)
        self.writedir = "."
        self.maxiter = 20
        self.tol = 1e-13
        self.dt = 1e10
        self.neq = 1
        self.nu = 1e-4
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)
        self.beta = np.ones_like(u)
        self.objective = TestObjective()
        
    def calc_residual(self, q):
        R = np.zeros_like(q)
        R[:] = self.calc_momentum_residual(q)
        return R
        
    def calc_momentum_residual(self, q):
        u = q[0:self.n]
        y = self.y
        uyy = diff2(y, u)
        R = self.beta*self.nu*uyy - self.dp/self.rho
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R
    def calc_delJ_delbeta(self, q, beta):
        n = np.size(beta)
        dbeta = 1e-20
        dJdbeta = np.zeros_like(beta)
        for i in range(n):
            beta[i] = beta[i] + 1j*dbeta
            F = self.objective.objective(q, beta)
            dJdbeta[i] = np.imag(F)/dbeta
            beta[i] = beta[i] - 1j*dbeta
        return dJdbeta

    def calc_delJ_delq(self, q, beta):
        n = np.size(q)
        dq = 1e-30
        dJdq = np.zeros_like(q)
        for i in range(n):
            q[i] = q[i] + 1j*dq
            F = self.objective.objective(q, beta)
            dJdq[i] = np.imag(F)/dq
            q[i] = q[i] - 1j*dq
        return dJdq

    def calc_psi(self, q, beta):
        dRdq = self.calc_residual_jacobian(q)
        dJdq = self.calc_delJ_delq(q, beta)
        psi = linalg.solve(dRdq.transpose(), -dJdq.transpose())
        return psi

    def calc_delR_delbeta(self, q):
        nb = np.size(self.beta)
        n = np.size(q)
        dbeta = 1e-30
        dRdbeta = np.zeros([n,nb], dtype=q.dtype)
        for i in range(nb):
            self.beta[i] = self.beta[i] + 1j*dbeta
            R = self.calc_residual(q)
            dRdbeta[:,i] = np.imag(R[:])/dbeta
            self.beta[i] = self.beta[i] - 1j*dbeta
        return dRdbeta

    def calc_sensitivity(self):
        q = self.q
        beta = self.beta
        psi = self.calc_psi(q, beta)
        delJdelbeta = self.calc_delJ_delbeta(q, beta)
        delRdelbeta = self.calc_delR_delbeta(q)
        dJdbeta = delJdelbeta + psi.transpose().dot(delRdelbeta)
        return dJdbeta
        
    def calc_residual_jacobian(self, q, dq=1e-25):
        n = np.size(q)
        dRdq = np.zeros([n, n], dtype=q.dtype)
        for i in range(n):
            q[i] = q[i] + 1j*dq
            R = self.calc_residual(q)
            dRdq[:,i] = np.imag(R[:])/dq
            q[i] = q[i] - 1j*dq
        return dRdq

    def calc_dt(self):
        return self.dt*np.ones(self.n)

    def step(self, q, dt):
        R = self.calc_residual(q)
        dRdq = self.calc_residual_jacobian(q)
        dt = self.calc_dt()
        A = np.zeros_like(dRdq)
        n = self.n
        for i in range(0, n):
            A[i,i] = 1./dt[i]
        A = A - dRdq
        dq = linalg.solve(A, R)
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dq, l2norm
        
    def boundary(self, q):
        pass

    def solve(self):
        q = np.copy(self.q)
        dt = self.dt
        for i in range(self.maxiter):
            dq, l2norm = self.step(q, dt)
            q[:] = q[:] + dq[:]
            self.boundary(q)
            print "Iteration: %i Norm: %1.2e"%(i, l2norm)
            self.save(q)
            if l2norm < self.tol:
                self.postprocess(q)
                break

        self.postprocess(q)
        self.q[:] = q[:]

    def plot(self):
        plt.figure(1)
        plt.plot(self.y, self.q[0:self.n], 'r-')
        plt.show()

    def postprocess(self, q):
        q = q.astype(np.float64)
        n = self.n
        u = q[0:n]
        self.utau = self.Retau*self.nu*2.0
        self.yp = self.y*self.utau/self.nu
        self.up = u/self.utau
        self.uap = self.analytic_solution()/self.utau

    def save(self, q):
        q = q.astype(np.float64)
        n = self.n
        u = q[0:n]
        np.savetxt("%s/u"%self.writedir, u)
 
    def analytic_solution(self):
        return self.dp/(2*self.nu*self.rho)*(self.y**2 - self.y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Retau", type=float, default=550.0, required=True, help="Reynolds number.")
    parser.add_argument("--dt", type=float, default=1.0, required=True, help="Solver time step.")
    parser.add_argument("--tol", type=float, default=1e-10, required=True, help="Solver convergence tolerance.")
    parser.add_argument("--maxiter", type=int, default=10, required=True, help="Solver max iteration.")
    parser.add_argument("--force_boundary", action="store_true", help="Force boundary.")
    args = parser.parse_args()

    Retau = args.Retau
    dt = args.dt
    tol = args.tol
    maxiter = args.maxiter
    force_boundary = args.force_boundary

    dirname ="base_solution"
    y, u = load_solution_laminar(dirname)
    Retau = Retau
    eqn = LaminarEquation(y, u, Retau)
    eqn.dt = dt
    eqn.tol = tol
    eqn.maxiter = maxiter
    eqn.force_boundary = force_boundary
    eqn.writedir = "solution"
    eqn.solve()
    dns = load_data()[0]
    
    plt.ioff()
    plt.figure(1)
    plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\omega$')
    plt.semilogx(eqn.yp[::5], eqn.uap[::5], 'bo', label=r'Analytic', mfc="white")
    plt.xlabel(r"$y^+$")
    plt.ylabel(r"$u^+$")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()
