
from spacecraft import *
from known_spacecraft import *
from dynamics import *
import numpy as np
import jax
import jax.numpy as jnp
from admm.inc_admm import *
from tqdm import tqdm

class Constellation:

    def __init__(self, known_satellites, rogue_satellites, neighbors, mu_prior, Sigma_prior, sim_time, dt=50):

        ''' CONSTANTS AND FUNCTIONS '''
        self.knowns = known_satellites
        self.rogues = rogue_satellites
        self.T = sim_time
        self.t = np.linspace(0, np.round(self.T/dt)*dt, int(np.round(self.T/dt)+1))
        self.neighbors = neighbors

        self.fd = discrete_orbital_dynamics(gravity_sim, dt)
        self.fd_jac = lambda mu : linearize_dynamics(self.fd, mu)
        self.measurement_models = []
        self.measurement_jacs = []
        for i in range(len(self.knowns)):
            self.measurement_models.append(self.knowns[i].measurement_model)
            self.measurement_jacs.append(self.knowns[i].measurement_jacobian)

        Q = np.diag(np.tile([5, 5, 5, 0.5, 0.5, 0.5], len(self.rogues)))
        self.estimator = IncADMM(self.fd, self.fd_jac, Q, self.knowns[0].R)

        ''' UPDATED IN GET_STATES_MEASUREMENTS '''
        self.rogues_rv = np.zeros((6*len(self.rogues), self.t.size))
        self.rogues_oe = np.zeros((6*len(self.rogues), self.t.size))
        self.knowns_rv = np.zeros((6*len(self.knowns), self.t.size))
        self.knowns_oe = np.zeros((6*len(self.knowns), self.t.size))

        self.measurements = np.zeros((len(self.knowns), 3*len(self.rogues), self.t.size))

        ''' UPDATED BY STATE ESTIMATOR '''
        self.mu_rv = np.zeros((len(self.knowns), 6*len(self.rogues), self.t.size))
        self.Sigma_rv = np.zeros((len(self.knowns), 6*len(self.rogues), 6*len(self.rogues), self.t.size))

        self.mu_oe = np.zeros((len(self.knowns), 6*len(self.rogues), self.t.size))
        self.Sigma_oe = np.zeros((len(self.knowns), 6*len(self.rogues), 6*len(self.rogues), self.t.size))

        ''' UPDATE WITH PRIOR '''
        self.collect_states_measurements(0)

        for i in range(len(self.knowns)):
            self.mu_rv[i,:,0] = mu_prior
            self.Sigma_rv[i,:,:,0] = Sigma_prior
            self.mu_oe[i,:,0], self.Sigma_oe[i,:,:,0] = propagate_eci_to_oe(mu_prior, Sigma_prior)



    def collect_states_measurements(self, k):

        # Get the rogue states
        for i in range(len(self.rogues)):
            sat = self.rogues[i]
            self.rogues_rv[6*i:6*i+3,k], self.rogues_rv[6*i+3:6*i+6,k] = sat.get_state_eci(self.t[k])
            self.rogues_oe[6*i:6*i+6,k] = np.array([sat.a, sat.e, sat.i, sat.Om, sat.w, sat.M])

        # Collect measurements
        for i in range(len(self.knowns)):
            sat = self.knowns[i]
            sat.collect_measurements(self.t[k])
            self.knowns_rv[6*i:6*i+3,k] = sat.r
            self.knowns_rv[6*i+3:6*i+6,k] = sat.v
            self.knowns_oe[6*i:6*i+6,k] = np.array([sat.a, sat.e, sat.i, sat.Om, sat.w, sat.M])
            self.measurements[i,:,k] = sat.y



    def run_sim(self):
        for k in tqdm(range(self.t.size-1)):
            # for l in range(len(self.rogues)):
            #     if np.random.rand() > 0.9:
            #         self.rogues[l].apply_control(np.array([np.random.rand()*0.5-0.25, np.random.rand()*0.5-0.25, np.random.rand()*0.5-0.25]), self.t[k+1])
            self.collect_states_measurements(k+1)
            mu_list, Sigma_list = self.estimator.solve(self.measurements[:,:,k+1], self.measurement_models, \
            self.measurement_jacs, self.mu_rv[:,:,k], self.Sigma_rv[:,:,:,k], self.neighbors)
            self.mu_rv[:,:,k+1] = np.array(mu_list)
            self.Sigma_rv[:,:,:,k+1] = np.array(Sigma_list)
            # for i in range(len(self.knowns)):
            #     self.mu_oe[i,:,k+1], self.Sigma_oe[i,:,:,k+1] = propagate_eci_to_oe(self.mu_rv[i,:,k+1], self.Sigma_rv[i,:,:,k+1])
