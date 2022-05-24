import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sys
sys.path.insert(0,"..")
from admm.admm import ADMM_Estimator
from tqdm import tqdm

def admm_step(mu0,cov0,y1,y2,A,C1,C2,Q,R,rho=10,tol=1e-3):
    n = len(mu0)
    m = len(y)
    p1 = np.zeros((n,1))
    p2 = np.zeros((n,1))
    z = np.linalg.inv(A@cov0@A.T + Q)
    Ri = np.linalg.inv(R)
    v1 = np.linalg.inv(C1.T@Ri@C1 + 0.5 * z + rho * np.eye(n))
    v2 = np.linalg.inv(C2.T@Ri@C2 + 0.5 * z + rho * np.eye(n))

    x1 = np.linalg.inv(C1.T@Ri@C1 + 0.5 * z) @ (C1.T@Ri@y1 + 0.5 * z@A@mu0)
    x2 = np.linalg.inv(C2.T@Ri@C2 + 0.5 * z) @ (C2.T@Ri@y1 + 0.5 * z@A@mu0)

    prev_p1 = np.ones((n,1))
    prev_p2 = np.ones((n,1))
    i = 1
    while True:
        p1 += rho * (x1-x2)
        p2 += rho * (x2-x1)

        x1_prev = x1.copy()
        x2_prev = x2.copy()

        x1 = v1 @ (C1.T@Ri@y1 + 0.5 * z@A@mu0 - 0.5 * p1 + 0.5 * rho * (x1_prev+x2_prev))
        x2 = v2 @ (C2.T@Ri@y2 + 0.5 * z@A@mu0 - 0.5 * p2 + 0.5 * rho * (x1_prev+x2_prev))

        d = np.linalg.norm(x1-x2)
        i += 1

        if d < tol:
            break

    cov1 = C1.T@Ri@C1 + z
    cov2 = C2.T@Ri@C2 + z
    return x1,x2,cov1,cov2

def admm(mu0,cov0,y,A,gs,fCs,Q,R,rho=10,tol=1e-3):

    n = len(mu0)
    T = y.shape[0]

    num_agents = len(gs)
    m = int(y.shape[1]/num_agents)

    mu = np.zeros((T,n))
    cov = np.zeros((T,n,n))
    mu[0,:] = mu0.T
    cov[0,:,:] = cov0

    for t in range(T-1):
        g1 = gs[0]
        C1 = fCs[0](A@mu[t,:].reshape(-1,1))
        g2 = gs[1]
        C2 = fCs[1](A@mu[t,:].reshape(-1,1))
        
        y1 = y[t+1,:m].reshape(-1,1) - g1(A@mu[t,:].reshape(-1,1)) + C1@A@mu[t,:].reshape(-1,1)
        y2 = y[t+1,m:].reshape(-1,1) - g2(A@mu[t,:].reshape(-1,1)) + C2@A@mu[t,:].reshape(-1,1)

        x1,x2,cov1,cov2 = admm_step(mu[t,:].reshape(-1,1),cov[t,:,:],y1,y2,A,C1,C2,Q,R,rho,tol)
        mu[t+1,:] = x1.T
        cov[t+1,:,:] = 0.5 * np.linalg.inv(cov1+cov2)
    return mu,cov

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

    for t in range(T-1):
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
tmax = 1
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


admm_est = ADMM_Estimator(gs,fCs,f_d,fA,neighbors,mu0,cov0,meas,Q,R_ind)
admm_est.step()
print(admm_est.x[1,:,:])

#mu,cov = ekf(mu0,cov0,y,A,g,fC,Q,R)
#plot_state_traj(t,x,mu,cov,"../figures/ekf.png")
#mu_admm,cov_admm = admm(mu0,cov0,y,A,gs,fCs,Q,R_ind,rho=1,tol=1e-3)
#plot_state_traj(t,x,mu_admm,cov_admm,"../figures/admm.png")
