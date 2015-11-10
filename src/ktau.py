import numpy as np
import matplotlib.pyplot as plt

from laminar import LaminarEquation
from utils import calc_dp
from schemes import diff, diff2

def get_var(q):
    n = np.size(q)
    ny = n/3
    u = q[0:n:3]
    k = q[1:n:3]
    tau = q[2:n:3]
    return u, k, tau

class KTauEquation(LaminarEquation):
    def __init__(self, y, u, k, tau, Retau):
        self.y = np.copy(y)
        ny = np.size(self.y)
        
        self.q = np.zeros(3*ny, dtype=np.complex)
        self.q[0:3*ny:3] = u[:]
        self.q[1:3*ny:3] = k[:]
        self.q[2:3*ny:3] = tau[:]
        self.Retau = Retau
        
        self.writedir = "."
        self.tol = 1e-11
        self.ny = ny
        self.n = self.ny*3
        
        self.maxiter = 1000
        self.dt = 1e6
        self.force_boundary = True
        
        self.neq = 1
        self.nu = 1e-4
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)

        self.sigma_k = 1.36
        self.cmu = 0.09
        self.sigma_tau2 = self.sigma_tau1 = self.sigma_k = 1.36
        self.Ceps1 = 1.44
        self.Ceps2 = 1.83
        self.A2 = 4.9


    def calc_momentum_residual(self, q):
        u, k, tau = get_var(q)
        y = self.y
        uy = diff(y, u)
        uyy = diff2(y, u)
        nut = self.calc_nut(q)
        nuty = diff(y, nut)
        R = self.nu*uyy - self.dp/self.rho
        R = R + nut*uyy + nuty*uy;
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R

    def calc_nut(self, q):
        cmu = self.cmu
        u, k, tau = get_var(q)
        utau = self.Retau*self.nu*2.0
        yp = utau*self.y/self.nu
        Ret = k*tau/self.nu
        fmu = (1 + 3.45/(np.sqrt(Ret) + 1e-16))*np.tanh(yp/70.0)
        # THIS FIX IS REQUIRED OTHERWISE FMU BECOMES ZERO AT THE WALL
        # AND LEADS TO NON DIFFERENTIABLE FMU
        fmu[0] = fmu[1]
        nut = cmu*k*tau*fmu
        return nut
        
    def calc_k_residual(self, q):
        nu = self.nu
        sigma_k = self.sigma_k
        y = self.y
        u, k, tau = get_var(q)
        uy = diff(self.y, u)
        ky = diff(self.y, k)
        kyy = diff2(self.y, k)
        tauy = diff(self.y, tau)
        nut = self.calc_nut(q)
        nuty = diff(self.y, nut)
        R = nut*uy**2 - k/(tau + 1e-10) + kyy*(nu + nut/sigma_k) + nuty*ky/sigma_k
        R[0] = -k[0]
        R[-1] = 1/(y[-1] - y[-2])*(1.5*k[-1] - 2*k[-2] + 0.5*k[-3])
        return R

    def calc_tau_residual(self, q):
        nu = self.nu
        sigma_tau2 = self.sigma_tau2
        sigma_tau1 = self.sigma_tau1
        Ceps1 = self.Ceps1
        Ceps2 = self.Ceps2
        A2 = self.A2

        y = self.y
        u, k, tau = get_var(q)
        uy = diff(self.y, u)
        ky = diff(self.y, k)
        tauy = diff(self.y, tau)
        tauyy = diff2(self.y, tau)
        nut = self.calc_nut(q)
        nuty = diff(self.y, nut)
        utau = self.Retau*self.nu*2.0
        yp = utau*self.y/self.nu
        Ret = k*tau/self.nu
        f2 = (1.0 - 2.0/9.0*np.exp(-Ret**2/36.0))*(1 - np.exp(-yp/A2))**2
        R = (1.0 - Ceps1)*tau/(k+1e-16)*nut*uy**2 + (Ceps2*f2-1) + tauyy*(nu + nut/sigma_tau2) + tauy*nuty/sigma_tau2 - 2.0/(tau + 1e-16)*(nu + nut/sigma_tau2)*tauy**2 + 2.0/(k+1e-16)*(nu + nut/sigma_tau1)*ky*tauy
        R[0] = -tau[0]
        R[-1] = 1/(y[-1] - y[-2])*(1.5*tau[-1] - 2.0*tau[-2] + 0.5*tau[-3])
        return R
        
    def calc_residual(self, q):
        n = self.n
        R = np.zeros_like(q)
        R[0:n:3] = self.calc_momentum_residual(q)
        R[1:n:3] = self.calc_k_residual(q)
        R[2:n:3] = self.calc_tau_residual(q)
        return R

    def calc_dt(self):
        dt = np.zeros(self.n)
        for i in range(0, self.n, 3):
            dt[i] = self.dt*1000
            dt[i+1] = self.dt
            dt[i+2] = self.dt
        return dt

    def boundary(self, q):
        if self.force_boundary:
            q[0] = 0.0
            q[1] = 0.0
            q[2] = 0.0
            q[-1] = q[-4]
            q[-2] = q[-5]
            q[-3] = q[-6]

    def postprocess(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, k, tau = get_var(q)
        self.utau = self.Retau*self.nu*2.0
        self.yp = self.y*self.utau/self.nu
        self.up = u/self.utau
        self.kp = k/self.utau**2
        self.taup = tau/(self.nu/self.utau**2)
        
    def save(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, k, tau = get_var(q)
        np.savetxt("%s/u"%self.writedir, u)
        np.savetxt("%s/k"%self.writedir, k)
        np.savetxt("%s/tau"%self.writedir, tau)

    def plot(self):
        plt.figure(1)
        plt.subplot(311)
        plt.semilogx(self.yp, self.up, 'r-')
        plt.subplot(312)
        plt.semilogx(self.yp, self.kp, 'r-')
        plt.subplot(313)
        plt.semilogx(self.yp, self.tauy, 'r-')
        plt.show()
        
