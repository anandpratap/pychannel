import numpy as np
from schemes import diff

class Data(object):
    def __init__(self):
        pass

def calc_dp(Retau, nu):
    utau = Retau*nu*2.0
    tauw = utau**2
    dp = -tauw*2.0
    return dp

def load_solution_komega(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.complex)
    u = np.loadtxt("%s/u"%dir_).astype(np.complex)
    k = np.loadtxt("%s/k"%dir_).astype(np.complex)
    omega = np.loadtxt("%s/omega"%dir_).astype(np.complex)
    return y, u, k, omega

def load_solution_ktau(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.complex)
    u = np.loadtxt("%s/u"%dir_).astype(np.complex)
    k = np.loadtxt("%s/k"%dir_).astype(np.complex)
    tau = np.loadtxt("%s/tau"%dir_).astype(np.complex)
    return y, u, k, tau

def load_solution_stressomega(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.complex)
    u = np.loadtxt("%s/u"%dir_).astype(np.complex)
    omega = np.loadtxt("%s/omega"%dir_).astype(np.complex)
    k = np.loadtxt("%s/k"%dir_).astype(np.complex)
    dudy = diff(y, u)
    nut = k/(omega + 1e-16)
    R11 = np.loadtxt("%s/R11"%dir_).astype(np.complex)[:201]
    R12 = np.loadtxt("%s/R12"%dir_).astype(np.complex)[:201]
    R22 = np.loadtxt("%s/R22"%dir_).astype(np.complex)[:201]
    R33 = np.loadtxt("%s/R33"%dir_).astype(np.complex)[:201]
    return y, u, R11, R12, R22, R33, omega

def load_solution_laminar(dir_):
    y = np.loadtxt("%s/y"%dir_).astype(np.complex)
    u = np.loadtxt("%s/u"%dir_).astype(np.complex)
    return y, u


def load_data():
    data = np.loadtxt("data/DNSsol.dat")
    y = data[:,0]
    yp = data[:,1]
    u = data[:,2]
    ub = data[:,3]
    vb = data[:,4]
    wb = data[:,5]
    k = 0.5*(ub**2 + vb**2 + wb**2)
    uv = data[:,10]
    
    kbal = np.loadtxt("data/kbal.dat")
    eps = kbal[:,2]
    kprod = kbal[:,3]

    DNS = Data()
    DNS.y = y
    DNS.yp = yp
    DNS.u = u
    DNS.ub = ub
    DNS.vb = vb
    DNS.wb = wb
    DNS.k = k 
    DNS.uv = uv
    DNS.eps = eps
    DNS.kprod = kprod
    
    wilcox = np.loadtxt("data/wilcox_sw.dat")
    yw = wilcox[:,2]
    uw = wilcox[:,3]
    uvw = -wilcox[:,4]
    kw = wilcox[:,5]
    ew = wilcox[:,6]
    pw = wilcox[:,7]
    Wilcox = Data()
    Wilcox.y = yw
    Wilcox.u = uw
    Wilcox.k = kw
    Wilcox.e = ew
    Wilcox.p = pw
    Wilcox.uv = uvw

    wilcox = np.loadtxt("data/wilcox.dat")
    yw = wilcox[:,2]
    uw = wilcox[:,3]
    uvw = -wilcox[:,4]
    kw = wilcox[:,5]
    ew = wilcox[:,6]
    pw = wilcox[:,7]
    Wilcox_kw = Data()
    Wilcox_kw.y = yw
    Wilcox_kw.u = uw
    Wilcox_kw.k = kw
    Wilcox_kw.e = ew
    Wilcox_kw.p = pw
    Wilcox_kw.uv = uvw

    return DNS, Wilcox, Wilcox_kw
