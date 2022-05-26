import numpy as np
import copy
from tqdm import tqdm

class ADMM_Estimator:
    def __init__(self,meas_funcs,meas_jacs,f_d,fA,neighbors,mu0,cov0,meas,Q,R,mu=.5,max_iter=100):
        """
            Constructor for the ADMM_Estimator class

            ==== Required Arguments ====
            meas_funcs: A list of measurement functions for each agent/sensor
            meas_jacs: A list of measurement jacobian functions for each agent/sensor
            f_d: Dynamics function of two arguements. x_k+1 = f_d(x_k,k)
            fA: Dynamics jacobian function of two arguments (x_k,k)
            neighbors: A list of lists. Indexing in by an agent's index gives a list of its neighbor indices
            mu0: numpy array of initial state estimate. Shape should be (n,)
            cov0: initial state estimate covariance. Shape should be (n,n)
            meas: numpy array of measurements. Shape should be TxNxm where:
                T = number of timesteps
                N = number of agents
                m = measurement dimension
            Q: dynamics noise covariance
            R: measurement noise covariance

            ==== Optional Arguments ====
            max_iter: Maximum Iterations for ADMM solves. Default = 100
        """
        self.meas_funcs = meas_funcs
        self.meas_jacs = meas_jacs
        self.f_d = f_d
        self.fA = fA
        self.neighbors = neighbors
        self.mu0 = mu0
        self.cov0 = cov0
        self.meas = meas
        self.Q = Q
        self.R = R
        self.Ri = np.linalg.inv(R)

        self.mu = mu
        self.max_iter = max_iter

        (self.T,self.N,self.m) = meas.shape
        self.n = mu0.shape[0]

        self.x = np.zeros((self.T,self.N,self.n))
        self.cov = np.zeros((self.T,self.N,self.n,self.n))

        for i in range(self.N):
            self.x[0,i,:] = mu0.flatten()
            self.cov[0,i,:,:] = cov0

        self.t = 0

    def step(self,tol=1e-3):
        """
            Runs a single round of ADMM iterations to convergence.
        """
        lam = np.zeros((self.N,self.N,self.n))
        zs = np.zeros((self.N,self.n))
        x_props = [self.f_d(self.x[self.t,i,:],self.t) for i in range(self.N)]
        for i in range(self.N):
            zs[i,:] = x_props[i]

        As = [self.fA(self.x[self.t,i,:],self.t) for i in range(self.N)]
        Cs = [self.meas_jacs[i](x_props[i]) for i in range(self.N)]
        ys = [self.meas[self.t,i,:] - self.meas_funcs[i](x_props[i]) + Cs[i]@x_props[i] for i in range(self.N)]
        cov_invs = [np.linalg.inv(As[i]@self.cov[self.t,i,:,:]@As[i].T + self.Q) for i in range(self.N)]

        invs = [np.linalg.inv(Cs[i].T@self.Ri@Cs[i] + 1/self.N * cov_invs[i] + 1 / self.mu * (len(self.neighbors[i])+1)*np.eye(self.n)) for i in range(self.N)]
        xs = [np.linalg.solve(Cs[i].T@self.Ri@Cs[i] + 1/self.N * cov_invs[i],(Cs[i].T@self.Ri@ys[i] + 1/self.N * cov_invs[i]@x_props[i]).reshape(-1,1)) for i in range(self.N)]

        # Do ADMM iterations
        for k in range(self.max_iter):

            prev_xs = copy.deepcopy(xs)
            prev_zs = copy.deepcopy(zs)
            prev_lam = copy.deepcopy(lam)

            # Primal solve
            for i in range(self.N):
                xs[i] = invs[i]@(Cs[i].T@self.Ri@ys[i] + 1/self.N * cov_invs[i]@x_props[i] + sum([zs[j]/self.mu+lam[i,j] for j in self.neighbors[i]+[i,]]))

            # Dual updates
            for i in range(self.N):
                zs[i,:] = self.mu / (len(self.neighbors[i])+1) * sum([1/self.mu * xs[j] - lam[j,i] for j in self.neighbors[i]+[i,]])

            for i in range(self.N):
                for j in self.neighbors[i]+[i,]:
                    lam[i,j] = lam[i,j] - 1/self.mu * (xs[i]-zs[j])

            # Check stopping criteria
            diff_xs = sum([np.linalg.norm(xs[i]-prev_xs[i]) for i in range(self.N)])
            diff_zs = sum([np.linalg.norm(zs[i]-prev_zs[i]) for i in range(self.N)])
            if diff_xs < tol and diff_zs < tol:
                break

        # Update means and covariances
        for i in range(self.N):
            self.x[self.t+1,i,:] = xs[i].flatten()
            self.cov[self.t+1,i,:,:] = np.linalg.inv(sum([Cs[j].T@self.Ri@Cs[j] + cov_invs[j] for j in self.neighbors[i]+[i,]]))

        self.t += 1

    def run(self):
        """
            Runs ADMM over all measurements
        """
        for _ in tqdm(range(self.T-1)):
            self.step()
