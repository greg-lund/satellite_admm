import numpy as np
import copy

class ADMM_Estimator:
    def __init__(self,meas_funcs,meas_jacs,f_d,fA,neighbors,mu0,cov0,meas,Q,R,penalty=20.0,max_iter=1000):
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
            penalty: ADMM penalty term (rho). Default = 1.0
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

        self.penalty = penalty
        self.max_iter = max_iter

        (self.T,self.N,self.m) = meas.shape
        self.n = mu0.shape[0]

        self.x = np.zeros((self.T,self.N,self.n))
        self.cov = np.zeros((self.T,self.N,self.n,self.n))

        self.t = 0

    def step(self,tol=1e-3):
        """
            Runs a single round of ADMM iterations to convergence.
        """
        x_props = [self.f_d(self.x[self.t,i,:],self.t) for i in range(self.N)]
        As = [self.fA(self.x[self.t,i,:],self.t) for i in range(self.N)]
        Cs = [self.meas_jacs[i](x_props[i]) for i in range(self.N)]
        ys = [self.meas[self.t,i,:] - self.meas_funcs[i](x_props[i]) + Cs[i]@x_props[i] for i in range(self.N)]
        cov_invs = [np.linalg.inv(As[i]@self.cov[self.t,i,:,:]@As[i].T + self.Q) for i in range(self.N)]

        invs = [np.linalg.inv(Cs[i].T@self.Ri@Cs[i] + 1/self.N * cov_invs[i] + self.penalty * len(self.neighbors[i])*np.eye(self.n)) for i in range(self.N)]
        ps = np.zeros((self.N,self.n))
        xs = [np.linalg.solve(Cs[i].T@self.Ri@Cs[i] + 1/self.N * cov_invs[i],(Cs[i].T@self.Ri@ys[i] + 1/self.N * cov_invs[i]@x_props[i]).reshape(-1,1)) for i in range(self.N)]

        # Do ADMM iterations
        for k in range(self.max_iter):

            ps_prev = copy.deepcopy(ps)
            xs_prev = copy.deepcopy(xs)

            # Dual Ascent
            for i in range(self.N):
                ps[i,:] = ps[i,:] + self.penalty * np.sum([xs[i] - xs[j] for j in self.neighbors[i]],axis=0).flatten()

            # Primal Minimize
            for i in range(self.N):
                xs[i] = (invs[i] @ (Cs[i].T@self.Ri@ys[i] + 1/self.N * cov_invs[i]@x_props[i] - 0.5 * ps[i,:] + 0.5 * self.penalty * np.sum([xs_prev[i]+xs_prev[j] for j in self.neighbors[i]],axis=0).flatten())).reshape(-1,1)

            # Check stopping criteria
            diff_xs = sum([np.linalg.norm(xs[i]-xs_prev[i]) for i in range(self.N)])
            diff_ps = sum([np.linalg.norm(ps[i]-ps_prev[i]) for i in range(self.N)])
            if diff_xs < tol and diff_ps < tol:
                print("ADMM step met stopping criteria in %d steps"%(k+1))
                break

        # Update means and covariances
        for i in range(self.N):
            self.x[self.t+1,i,:] = xs[i].flatten()
            self.cov[self.t+1,i,:,:] = np.linalg.inv(np.sum([Cs[j].T@self.Ri@Cs[j] + cov_invs[j] for j in self.neighbors[i]],axis=0))

        self.t += 1
