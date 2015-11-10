import numpy as np

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
    DNS.k = k 
    DNS.uv = uv
    DNS.eps = eps
    DNS.kprod = kprod
    
    wilcox = np.loadtxt("data/wilcox.dat")
    yw = wilcox[:,2]
    uw = wilcox[:,3]
    kw = wilcox[:,5]
    ew = wilcox[:,6]
    pw = wilcox[:,7]
    Wilcox = Data()
    Wilcox.y = yw
    Wilcox.u = uw
    Wilcox.k = kw
    Wilcox.e = ew
    Wilcox.p = pw
    return DNS, Wilcox
