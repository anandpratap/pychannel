import matplotlib.pyplot as plt
import numpy as np

from utils import calc_dp
from schemes import diff, diff2

class LaminarEquation(object):
    def __init__(self, y, u, Retau):
        self.y = np.copy(y)
        self.q = np.copy(u.astype(np.complex))
        self.Retau = Retau

        #
        self.n = np.size(y)
        self.writedir = "."
        self.maxiter = 10
        self.tol = 1e-13
        self.dt = 1e10
        self.neq = 1
        self.nu = 1e-4
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)

    def calc_residual(self, q):
        R = np.zeros_like(q)
        R[:] = self.calc_momentum_residual(q)
        return R
        
    def calc_momentum_residual(self, q):
        u = q[0:self.n]
        y = self.y
        uyy = diff2(y, u)
        R = self.nu*uyy - self.dp/self.rho
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R

    def calc_residual_jacobian(self, q, dq=1e-20):
        n = np.size(q)
        dRdq = np.zeros([n, n], dtype=np.float64)
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
        dq = np.linalg.solve(A, R.astype(np.float64))
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

