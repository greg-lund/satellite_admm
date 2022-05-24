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
    xlbl = ["x","y","vx","vy"]
    mulbl = ["mu_x","mu_y","mu_vx","mu_vy"]
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

np.random.seed(1)
dt = 1e-1
tmax = 10
t = np.arange(0,tmax,dt)
n = 4
m = 2

A = np.vstack([np.hstack([np.eye(2),dt*np.eye(2)]),np.hstack([np.zeros((2,2)),np.eye(2)])])
Q = 0.1 * dt * np.eye(n)
R = 0.1 * dt * np.eye(m)
R_ind = 0.1 * dt * np.eye(1)

C = np.hstack([np.vstack([np.eye(2),np.eye(2)]),np.zeros((4,2))])

x0 = np.array([0,0,1,1]).reshape(-1,1)

# Sensor locations
s1 = np.array([0,10]).reshape(-1,1)
s2 = np.array([10,0]).reshape(-1,1)

x = np.zeros((len(t),n))
y = np.zeros((len(t),m))
meas = np.zeros((len(t),2,1))
x[0,:] = x0.T
for i in range(len(t)-1):
    w = np.random.randn(n)
    xn = A @ x[i,:] + scipy.linalg.sqrtm(Q) @ w
    x[i+1,:] = xn.reshape(n,)
    y[i+1,:] = np.array([rng_meas(s1,x[i+1,:]),rng_meas(s2,x[i+1,:])]) + scipy.linalg.sqrtm(R) @ np.random.randn(m)
    meas[i,0,:] = y[i+1,0]
    meas[i,1,:] = y[i+1,1]

mu0 = np.array([0,0,0,0]).reshape(-1,1)
cov0 = np.eye(4)

g = lambda x: np.vstack([rng_meas(s1,x),rng_meas(s2,x)])
fC = lambda x: np.vstack([rng_jac(s1,x),rng_jac(s2,x)])

gs = [lambda x: rng_meas(s1,x),lambda x: rng_meas(s2,x)]
fCs = [lambda x: rng_jac(s1,x),lambda x: rng_jac(s2,x)]
f_d = lambda x,t: A@x
fA = lambda x,t: A
neighbors = [[1],[0]]


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
