import numpy as np
import copy
from tqdm import tqdm

class IncADMM:
    def __init__(self,f,fA,Q,R,penalty=20.0,max_iter=100,tol=1e-5):
        """
            Constructor for the Incremental ADMM Estimator class

            ==== Required Arguments ====
            f: Dynamics function of one arguement. x_k+1 = f_d(x_k) returns a jnp array
            fA: Dynamics jacobian function of one argument A_k = fA(x_k) returns a jnp array
            Q: dynamics noise covariance as a (n,n) numpy array
            R: measurement noise covariance as a (m,m) numpy array

            ==== Optional Arguments ====
            penalty: ADMM penalty term (rho). Default = 1.0
            max_iter: Maximum iterations for ADMM solves. Default = 100
            tol: Tolerance stopping criteria for ADMM solves. Default = 1e-5
        """
        self.f = f
        self.fA = fA
        self.Q = Q
        self.R = R
        self.Ri = np.linalg.inv(R)

        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol

    def central_step(self):
        """
        Run a centralized MAP estimate step. For debugging purposes
        """
        # Build correct size R matrix
        R = np.kron(np.eye(self.N),self.R)
        Ri = np.linalg.inv(R)

        A = self.fA(self.x[self.t,0,:],self.t)
        x_prop = self.f_d(self.x[self.t,0,:],self.t)
        cov_prop = A@self.cov[self.t,0,:,:]@A.T + self.Q
        cov_inv = np.linalg.inv(cov_prop)

        C = np.vstack([self.meas_jacs[i](x_prop) for i in range(self.N)])
        y = np.vstack([self.meas[self.t+1,i,:] for i in range(self.N)])
        y_exp = np.vstack([self.meas_funcs[i](x_prop) for i in range(self.N)])
        y_hat = (y - y_exp + C@x_prop.reshape(-1,1)).flatten()

        self.x[self.t+1,0,:] = np.linalg.inv(cov_inv + C.T@Ri@C)@(cov_inv@x_prop + C.T@Ri@y_hat)
        self.cov[self.t+1,0,:,:] = np.linalg.inv(C.T@Ri@C+cov_inv)


    def solve(self,y,g,fC,mu,cov,neighbors):
        """
            Runs a single round of ADMM iterations to convergence.
        """
        N = len(neighbors)
        n = len(mu[0])

        x_props = [self.f(mu[i]) for i in range(N)]
        As = [self.fA(mu[i]) for i in range(N)]
        Cs = [fC[i](x_props[i]) for i in range(N)]
        # print(g[0](x_props[0]))
        # print(Cs[0]@x_props[0])
        ys = [y[i] - g[i](x_props[i]) + Cs[i]@x_props[i] for i in range(N)]
        cov_invs = [np.linalg.inv(As[i]@cov[i]@As[i].T + self.Q) for i in range(N)]

        invs = [np.linalg.inv(Cs[i].T@self.Ri@Cs[i] + 1/N * cov_invs[i] + self.penalty * len(neighbors[i])*np.eye(n)) for i in range(N)]
        ps = np.zeros((N,n))
        xs = [np.linalg.solve(Cs[i].T@self.Ri@Cs[i] + 1/N * cov_invs[i],(Cs[i].T@self.Ri@ys[i] + 1/N * cov_invs[i]@x_props[i]).reshape(-1,1)) for i in range(N)]

        # Do ADMM iterations
        for k in range(self.max_iter):

            ps_prev = copy.deepcopy(ps)
            xs_prev = copy.deepcopy(xs)

            # Dual Ascent
            for i in range(N):
                ps[i,:] = ps[i,:] + self.penalty * np.sum([xs[i] - xs[j] for j in neighbors[i]],axis=0).flatten()

            # Primal Minimize
            for i in range(N):
                xs[i] = (invs[i] @ (Cs[i].T@self.Ri@ys[i] + 1/N * cov_invs[i]@x_props[i] - 0.5 * ps[i,:] + 0.5 * self.penalty * np.sum([xs_prev[i]+xs_prev[j] for j in neighbors[i]],axis=0).flatten())).reshape(-1,1)

            # Check stopping criteria
            diff_xs = sum([np.linalg.norm(xs[i]-xs_prev[i]) for i in range(N)])
            diff_ps = sum([np.linalg.norm(ps[i]-ps_prev[i]) for i in range(N)])
            if diff_xs < self.tol and diff_ps < self.tol:
                #print("ADMM step met stopping criteria in %d steps"%(k+1))
                break

        mu_hat,cov_hat = [],[]
        # Update means and covariances
        for i in range(N):
            mu_hat.append(xs[i].flatten())
            cov_hat.append(np.linalg.inv(np.sum([Cs[j].T@self.Ri@Cs[j] + 1/(len(neighbors[i])+1)*cov_invs[j] for j in neighbors[i]+[i,]],axis=0)))

        return mu_hat,cov_hat
