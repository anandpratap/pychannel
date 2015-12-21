from pylab import *
def gradedSpace(x0,xF,totalPoints,cell_rat):
    totalLength = xF-x0
    nx = totalPoints
    x= zeros(totalPoints,dtype='complex')
    dx = zeros(totalPoints-1,dtype='complex')
    if (cell_rat == 1.0):
        r = 1
        dx0 = totalLength/(totalPoints-1.)
    else:
        r = cell_rat**(1./(nx-2))
        L = totalLength
        dx0 = L/((1-r**(nx-1))/(1-r))
    x[0] = x0
    for i in range(1,nx):
        dx[i-1] = dx0*r**float(i-1)
        x[i] = x[i-1]+dx[i-1]
        
    dxEnd = x[nx-1] - x[nx-3]
    x[nx-2] = x[nx-3] + 0.5*dxEnd
    return x


if __name__ == "__main__":
    Retau = 180.0
    nu = 1e-4
    utau = Retau*nu*2.0
    yp = 1.0
    y_1 = yp*nu/utau
    n = 41
    y = gradedSpace(0.0, 0.5, n, 400.0)
    print real(y[1]), "should be less than", y_1
    plot(y, y, 'x-')
    show()
    np.savetxt("y_%i"%n, y.astype(np.float64))
