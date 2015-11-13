import numpy as np
import matplotlib.pyplot as plt

class InverseSolver(object):
    def __init__(self, eqn):
        self.eqn = eqn
        self.maxiter = 1000
        self.dostats = False
        self.algo = "sd"
        self.stepsize = 0.1
    def sample_prior(self):
        pass
    
    def sample_posterior(self):
        pass

    def calc_stats(self):
        pass

    def get_sensitivity(self):
        pass

    def get_stepsize(self):
        return 1.0

    def step_sd(self):
        self.eqn.solve()
        dJdbeta = self.eqn.calc_sensitivity()
        dJdbeta_norm= dJdbeta/np.linalg.norm(dJdbeta)
        pk = -dJdbeta_norm
        stepsize = self.linesearch(self.stepsize, pk)
        self.eqn.beta += stepsize*pk
    
    def step_bfgs(self):
        self.eqn.solve()
        dJdbeta = self.eqn.calc_sensitivity()
        dJdbeta_norm= dJdbeta/np.linalg.norm(dJdbeta)
        
        if self.i == 0:
            self.B = np.eye(np.size(self.eqn.beta))
            
        else:
            yk = (dJdbeta_norm - self.dJdbeta_norm)[np.newaxis].T
            sk = self.sk[np.newaxis].T
            term_1_num = yk.dot(yk.transpose())
            term_1_den = yk.transpose().dot(sk)
            term_2_num = self.B.dot(sk.dot(sk.transpose().dot(self.B)))
            term_2_den = sk.transpose().dot(self.B.dot(sk))
            self.B = self.B + term_1_num/term_1_den - term_2_num/term_2_den

        pk = np.linalg.solve(self.B, -dJdbeta_norm)
        pk = pk/np.linalg.norm(pk)
        stepsize = self.linesearch(1.1, pk)
        sk = stepsize*pk
        self.eqn.beta += sk
        self.sk = sk
        self.dJdbeta_norm = dJdbeta_norm

    def linesearch(self, stepsize, pk):
        beta_ = self.eqn.beta.copy()
        J_ = self.eqn.objective.objective(self.eqn.q, self.eqn.beta)
        for i in range(10):
            self.eqn.beta = beta_ + pk*stepsize
            self.eqn.solve()
            J = self.eqn.objective.objective(self.eqn.q, self.eqn.beta)
            if J < J_:
                self.eqn.beta[:] = beta_[:]
                break
            else:
                stepsize /= 2.0
        self.eqn.beta[:] = beta_[:]
        return stepsize


    def calculate_hessian(self):
        pass

    def calculate_cholesky(self):
        H = self.calculate_hessian()
        Cov = np.linalg.inv(H)
        R = np.chol(Cov)
        return R
        
    def solve(self):
        if self.dostats:
            self.sample_prior()

        for i in range(self.maxiter):
            self.i = i
            if self.algo == "sd":
                self.step_sd()
            else:
                self.step_bfgs()
            J = self.eqn.objective.objective(self.eqn.q, self.eqn.beta)
            print 30*"#", "ITER: ", i, "J: ", J

            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.semilogx(self.eqn.yp, self.eqn.q[::6], "r-", label="Inverse")
            plt.semilogx(self.eqn.yp, self.eqn.objective.val_target, label="DNS")
            plt.legend(loc=2)
            plt.pause(0.0001)
            
        if self.dostats:
            R = self.calculate_cholesky()
            self.sample_posterior()

        return self.eqn
            
        
if __name__ == "__main__":
    pass
