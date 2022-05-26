import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from tqdm import tqdm 

import sys
sys.path.insert(0,"..")
from admm.admm import ADMM_Estimator

def rng_meas(s,x):
    return np.linalg.norm(s.flatten()-x[:2].flatten())

def rng_jac(s,x):
    v = x[:2].reshape(-1,1) - s.reshape(-1,1)
    return np.hstack([v.T,np.zeros((1,2))]) / np.linalg.norm(v)

def ekf(mu0,cov0,y,A,g,fC,Q,R):
    n = len(mu0)
    m = y.shape[1]
    T = len(y)
    mu = np.zeros((T,n))
    cov = np.zeros((T,n,n))

    mu[0,:] = mu0.T
    cov[0,:,:] = cov0

    for t in tqdm(range(T-1)):
        # Predict
        mu[t+1,:] = A @ mu[t,:]
        cov[t+1,:,:] = A @ cov[t,:,:] @ A.T + Q

        # Update
        C = fC(mu[t+1,:])
        k = cov[t+1,:,:] @ C.T @ np.linalg.inv(C @ cov[t+1,:,:] @ C.T + R)
        mu[t+1,:] += k @ (y[t+1,:] - g(mu[t+1,:]).flatten())
        cov[t+1,:,:] -= k @ C @ cov[t+1,:,:]

    return mu,cov

def plot_state_traj(t,x,mu,cov,filename=None):
    fig,axs = plt.subplots(4)
    cmap = plt.get_cmap("tab10")
    xlbl = [r"$x$",r"$y$",r"$v_x$",r"$v_y$"]
    mulbl = [r"$\mu_x$",r"$\mu_y$",r"$\mu_{v_x}$",r"$\mu_{v_y}$"]
    for i in range(4):
        axs[i].plot(t,x[:,i],label=xlbl[i],color=cmap(0))
        axs[i].plot(t,mu[:,i],label=mulbl[i],color=cmap(1))
        axs[i].fill_between(t,mu[:,i]+1.96*np.sqrt(cov[:,i,i]),mu[:,i]-np.sqrt(cov[:,i,i]),color=cmap(1),alpha=0.2)
        axs[i].legend()
        axs[i].set_xlabel("Time [s]")

    if filename:
        plt.savefig(filename,dpi=300)
    else:
        plt.show()

#np.random.seed(1)
dt = 1e-1
tmax = 10
t = np.arange(0,tmax,dt)
n = 4

# Sensor locations
s = [np.array([0,1]).reshape(-1,1),np.array([1,0]).reshape(-1,1),np.array([2,-2]).reshape(-1,1),np.array([5,6]).reshape(-1,1)]
m = len(s)

A = np.vstack([np.hstack([np.eye(2),dt*np.eye(2)]),np.hstack([np.zeros((2,2)),np.eye(2)])])
Q = 0.1 * dt * np.eye(n)
R = 0.1 * dt * np.eye(m)
R_ind = 0.1 * dt * np.eye(1)

x0 = np.array([0,0,1,1]).reshape(-1,1)

x = np.zeros((len(t),n))
y = np.zeros((len(t),m))
meas = np.zeros((len(t),m,1))
x[0,:] = x0.T
for i in range(len(t)-1):
    w = np.random.randn(n)
    xn = A @ x[i,:] + scipy.linalg.sqrtm(Q) @ w
    x[i+1,:] = xn.reshape(n,)
    y[i+1,:] = np.array([rng_meas(s[j],x[i+1,:]) for j in range(m)]) + scipy.linalg.sqrtm(R) @ np.random.randn(m)
    for j in range(m):
        meas[i,j,:] = y[i+1,j]

mu0 = np.array([0,0,0,0]).reshape(-1,1)
cov0 = np.eye(4)

g = lambda x: np.vstack([rng_meas(s[i],x) for i in range(m)])
fC = lambda x: np.vstack([rng_jac(s[i],x) for i in range(m)])

gs = [lambda x,i=i: rng_meas(s[i],x) for i in range(m)]
fCs = [lambda x,i=i: rng_jac(s[i],x) for i in range(m)]
f_d = lambda x,t: A@x
fA = lambda x,t: A

# Just have fully connected ADMM
neighbors = [[j for j in [x for x in range(m) if x != i]] for i in range(m)]
print("neighbors:",neighbors)

print("Running ADMM")
admm_est = ADMM_Estimator(gs,fCs,f_d,fA,neighbors,mu0,cov0,meas,Q,R_ind)
admm_est.run()
mu_admm = admm_est.x[:,0,:]
cov_admm = admm_est.cov[:,0,:,:]

print("Running EKF")
mu,cov = ekf(mu0,cov0,y,A,g,fC,Q,R)

print("Saving plots")
plot_state_traj(t,x,mu,cov,"../figures/ekf.png")
plot_state_traj(t,x,mu_admm,cov_admm,"../figures/admm.png")
