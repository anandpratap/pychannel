import numpy as np
import matplotlib.pyplot as plt

from laminar import LaminarEquation
from utils import calc_dp
from schemes import diff, diff2

def get_var(q):
    n = np.size(q)
    ny = n/6
    u = q[0:n:6]
    R11 = q[1:n:6]
    R12 = q[2:n:6]
    R22 = q[3:n:6]
    R33 = q[4:n:6]
    omega = q[5:n:6]
    return u, R11, R12, R22, R33, omega

class StressOmegaEquation(LaminarEquation):
    def __init__(self, y, u, R11, R12, R22, R33, omega, Retau):
        self.y = np.copy(y)
        ny = np.size(self.y)
        
        self.q = np.zeros(6*ny, dtype=np.complex)
        self.q[0:6*ny:6] = u[:]
        self.q[1:6*ny:6] = R11[:]
        self.q[2:6*ny:6] = R12[:]
        self.q[3:6*ny:6] = R22[:]
        self.q[4:6*ny:6] = R33[:]
        self.q[5:6*ny:6] = omega[:]
        self.Retau = Retau
        
        self.writedir = "."
        self.tol = 1e-11
        self.ny = ny
        self.n = self.ny*6
        
        self.maxiter = 1000
        self.dt = 1e6
        self.force_boundary = False
        
        self.neq = 6
        self.nu = 1e-4
        self.rho = 1.0
        self.dp = calc_dp(self.Retau, self.nu)

    def calc_momentum_residual(self, q):
        u, R11, R12, R22, R33, omega = get_var(q)
        y = self.y
        uy = diff(y, u)
        uyy = diff2(y, u)
        R = self.nu*uyy - self.dp/self.rho
        R12y = diff(y, R12)
        R = R + R12y
        R[0] = -u[0]
        R[-1] = (1.5*u[-1] - 2.0*u[-2] + 0.5*u[-3])/(y[-1] - y[-2])
        return R

    def calc_prod_dest_pi(self, q):
        C1 = 9./5.
        C2 = 10./19.
        alpha_h = (8. + C2)/11.
        alpha = 13.0/25.0
        sigma = 0.5
        beta_0 = 0.0708
        beta_star = 9.0/100.0
        sigma_star = 0.6
        sigma_do = 1.0/8.0
        beta_0 = 0.0708
        alpha_hat = (8.0 + C2)/11.0
        beta_hat = (8.0*C2 - 2.0)/11.0
        gamma_hat = (60.0*C2 - 4.0)/55.0
        
        nu = self.nu
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        P11 = 2.0*R12*diff(y, u)
        P12 = R22*diff(y, u)
        P22 = np.zeros_like(u)
        P33 = np.zeros_like(u)

        D11 = np.zeros_like(u)
        D12 = R11*diff(y, u)
        D22 = 2.0*R12*diff(y, u)
        D33 = np.zeros_like(u)

        S11 = np.zeros_like(u)
        S12 = 0.5*diff(y, u)
        S22 = np.zeros_like(u)
        S33 = np.zeros_like(u)
        
        k = -0.5*(R11 + R22 + R33)
        P_trace = 0.5*(P11 + P12 + P22)
        S_trace = (S11 + S12 + S22)
        
        PI11 = beta_star*C1*omega*(R11 + 2.0/3.0*k) - alpha_hat*(P11 - 2.0/3.0*P_trace) - beta_hat*(D11 - 2.0/3.0*P_trace) - gamma_hat*k*(S11 - 1.0/3.0*S_trace)
        PI12 = beta_star*C1*omega*R12 - alpha_hat*P12 - beta_hat*D12 - gamma_hat*k*S12
        PI22 = beta_star*C1*omega*(R22 + 2.0/3.0*k) - alpha_hat*(P22 - 2.0/3.0*P_trace) - beta_hat*(D22 - 2.0/3.0*P_trace) - gamma_hat*k*(S22 - 1.0/3.0*S_trace)
        PI33 = beta_star*C1*omega*(R33 + 2.0/3.0*k) - alpha_hat*(P33 - 2.0/3.0*P_trace) - beta_hat*(D33 - 2.0/3.0*P_trace) - gamma_hat*k*(S33 - 1.0/3.0*S_trace)
        
        nut = k/(omega + 1e-16)
        
        eps_hat = 2.0/3.0*beta_star*omega*k
        eps11 = 1.0/4.0*nu*diff2(y, -R11)
        eps12 = 1.0/4.0*nu*diff2(y, -R12)
        eps22 = 1.0/4.0*nu*diff2(y, -R22)
        eps33 = 1.0/4.0*nu*diff2(y, -R33)

        self.P11 = P11
        self.P12 = P12
        self.P22 = P22
        self.P33 = P33
    
        self.D11 = D11
        self.D12 = D12
        self.D22 = D22
        self.D33 = D33
        
        self.PI11 = PI11
        self.PI12 = PI12
        self.PI22 = PI22
        self.PI33 = PI33

        self.eps_hat = eps_hat
        self.eps11 = eps11
        self.eps12 = eps12
        self.eps22 = eps22
        self.eps33 = eps33
        # this might be dangerous
        self.k = k
        self.nut = nut
        self.nuty = diff(y, self.nut)
        

    def calc_R11_residual(self, q):
        sigma_star = 0.6
        nu = self.nu
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        R = np.zeros_like(u)
        R11y = diff(y, R11)
        R11yy = diff2(y, R11)
        R = -self.P11 + self.eps_hat + self.eps11 - self.PI11 + R11yy*(self.nu + self.nut*sigma_star) + sigma_star*self.nuty*R11y
        R[0] = -R11[0]
        R[-1] = (1.5*R11[-1] - 2.0*R11[-2] + 0.5*R11[-3])/(y[-1] - y[-2])
        return R

    def calc_R12_residual(self, q):
        sigma_star = 0.6
        nu = self.nu
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        R = np.zeros_like(u)
        R12y = diff(y, R12)
        R12yy = diff2(y, R12)
        R = -self.P12 + self.eps12 - self.PI12 + R12yy*(self.nu + self.nut*sigma_star) + sigma_star*self.nuty*R12y
        R[0] = -R12[0]
        R[-1] = -R12[-1]
        return R

    def calc_R22_residual(self, q):
        sigma_star = 0.6
        nu = self.nu
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        R = np.zeros_like(u)
        R22y = diff(y, R22)
        R22yy = diff2(y, R22)
        R = -self.P22 + self.eps_hat + self.eps22 - self.PI22 + R22yy*(self.nu + self.nut*sigma_star) + sigma_star*self.nuty*R22y
        R[0] = -R22[0]
        R[-1] = (1.5*R22[-1] - 2.0*R22[-2] + 0.5*R22[-3])/(y[-1] - y[-2])
        return R

    def calc_R33_residual(self, q):
        sigma_star = 0.6
        nu = self.nu
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        R = np.zeros_like(u)
        R33y = diff(y, R33)
        R33yy = diff2(y, R33)
        R = -self.P33 + self.eps_hat + self.eps33 - self.PI33 + R33yy*(self.nu + self.nut*sigma_star) + sigma_star*self.nuty*R33y
        R[0] = -R33[0]
        R[-1] = (1.5*R33[-1] - 2.0*R33[-2] + 0.5*R33[-3])/(y[-1] - y[-2])
        return R

    def calc_omega_residual(self, q):
        nu = self.nu
        alpha = 13.0/25.0
        sigma = 0.5
        beta_0 = 0.0708
        y = self.y
        u, R11, R12, R22, R33, omega = get_var(q)
        k = -0.5*(R11 + R22 + R33)
        nut = k/(omega + 1e-16)
        nuty = diff(self.y, nut)
        uy = diff(self.y, u)
        ky = diff(self.y, k)
        omegay = diff(self.y, omega)
        omegayy = diff2(self.y, omega)
        lastterm = 1/(omega + 1e-16)*ky*omegay
        R = alpha*omega/(k + 1e-16)*R12*diff(y, u) - beta_0*omega**2 + omegayy*(nu + nut*sigma) + sigma*nuty*omegay + np.maximum(lastterm/8.0, 0.0)
        R[0] = -(omega[0] - 5000000*nu/0.005**2)
        R[-1] = 1/(y[-1] - y[-2])*(1.5*omega[-1] - 2.0*omega[-2] + 0.5*omega[-3])
        return R
        
    def calc_residual(self, q):
        n = self.n
        R = np.zeros_like(q)
        self.calc_prod_dest_pi(q)
        R[0:n:6] = self.calc_momentum_residual(q)
        R[1:n:6] = self.calc_R11_residual(q)
        R[2:n:6] = self.calc_R12_residual(q)
        R[3:n:6] = self.calc_R22_residual(q)
        R[4:n:6] = self.calc_R33_residual(q)
        R[5:n:6] = self.calc_omega_residual(q)
        return R

    def calc_dt(self):
        dt = np.zeros(self.n)
        for i in range(0, self.n, 6):
            dt[i] = self.dt*100000
            dt[i+1] = self.dt*10
            dt[i+2] = self.dt*10
            dt[i+3] = self.dt*10
            dt[i+4] = self.dt*10
            dt[i+5] = self.dt
        return dt

    def boundary(self, q):
        if self.force_boundary:
            q[0:5] = 0.0
            q[5] = 5000000*self.nu/0.005**2
            q[-1] = q[-7]
            q[-2] = q[-8]
            q[-3] = q[-9]
            q[-4] = 0.0
            q[-5] = q[-11]
            q[-6] = q[-12]
            self.plot(q)
            
    def postprocess(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, R11, R12, R22, R33, omega = get_var(q)
        self.utau = self.Retau*self.nu*2.0
        self.yp = self.y*self.utau/self.nu
        self.up = u/self.utau
        k = -0.5*(R11 + R22 + R33)
        self.kp = k/self.utau**2
        self.omegap = omega*self.nu/self.utau**2
        self.R11 = R11/self.utau**2
        self.R12 = R12/self.utau**2
        self.R22 = R22/self.utau**2
        self.R33 = R33/self.utau**2

    def save(self, q):
        q = q.astype(np.float64)
        n = self.n
        u, R11, R12, R22, R33, omega = get_var(q)
        np.savetxt("%s/u"%self.writedir, u)
        np.savetxt("%s/R11"%self.writedir, R11)
        np.savetxt("%s/R12"%self.writedir, R12)
        np.savetxt("%s/R22"%self.writedir, R22)
        np.savetxt("%s/R33"%self.writedir, R33)
        np.savetxt("%s/omega"%self.writedir, omega)

    def plot(self, q):
        u, R11, R12, R22, R33, omega = get_var(q)
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(331)
        plt.semilogx(self.y[1:], u[1:], 'r-')
        plt.subplot(332)
        plt.semilogx(self.y[1:], R11[1:], 'r-')
        plt.subplot(333)
        plt.semilogx(self.y[1:], R12[1:], 'r-')
        plt.subplot(334)
        plt.semilogx(self.y[1:], R22[1:], 'r-')
        plt.subplot(335)
        plt.semilogx(self.y[1:], R33[1:], 'r-')
        plt.subplot(336)
        plt.semilogx(self.y[1:], omega[1:], 'r-')
        plt.pause(0.0001)
#        plt.show()
        
