
from spacecraft import *
import jax
import jax.numpy as jnp
import numpy as np

class KnownSpacecraft(Spacecraft):

    def __init__(self, a, e, i, Om, w, M_start, knowns, rogues, mu_prior, Sigma_prior):

        super().__init__(a, e, i, Om, w, M_start)
        self.knowns = knowns
        self.rogues = rogues

        self.mu = mu_prior
        self.Sigma = Sigma_prior

        self.y = jnp.zeros(6 + 3*len(rogues))
        self.y_with_v = jnp.zeros(6 + 6*len(rogues))

        self.dt = 1

        covs = np.zeros(6 + 3*len(rogues))
        covs[0:6] = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        covs[6:] = np.array(0.1 * np.ones(3*len(rogues)))
        self.R = jnp.diag(covs)

        self.mu_history = []
        self.Sigma_history = []
        self.time_history = []

        self.mu_history.append(self.mu)
        self.Sigma_history.append(self.Sigma)
        self.time_history.append(0.)

        self.compile_estimator()



    def run_timestep(self, t):
        self.collect_measurements(t)
        mu, self.Sigma = self.estimate_state_fast(self.mu, self.Sigma, self.y)
        if jnp.isnan(mu).any():
            print(self.mu_history[-1])
            print(mu)
            raise ValueError("Bad Values!")
        # R_earth = 6371
        # for i in range(len(self.rogues)+1):
        #     if mu[6*i] < R_earth:
        #         mu[6*i] = R_earth
        #     if mu[6*i+1] < 0:
        #         mu[6*i+1] = 0
        #     # if mu[6*i+5] < 0:
        #     #     mu[6*i+5] = 0
        #     mu[6*i+5] = mu[6*i+5] % (2*jnp.pi)
        self.mu = mu
        self.mu_history.append(self.mu)
        self.Sigma_history.append(self.Sigma)
        self.time_history.append(t)



    def estimate_state(self, mu, Sigma, y):
        # Predict
        A = self.dynamics_jacobian(mu)
        mu = self.dynamics_model(mu)
        Sigma = A @ Sigma @ A.T
        # Update
        C = self.measurement_jacobian(mu)
        g = self.measurement_model(mu)
        K = Sigma @ C.T @ jnp.linalg.inv(C @ Sigma @ C.T + self.R)

        mu = mu + K @ (y - g)
        Sigma = Sigma - K @ C @ Sigma
        return (mu, Sigma)



    def compile_estimator(self):
        self.estimate_state_fast = jax.jit(self.estimate_state)



    def check_satellite_accessible(self, satellite, t):

        # Get the relative positition between the two satellites
        r_sat, v_sat = satellite.get_state_eci(t)
        r, v = self.get_state_eci(t)
        rel_pos = r_sat - r
        rel_vel = v_sat - v

        R_earth = 6371 # Radius of the Earth

        # Find the angle describing the cone that circumscribes the Earth
        r_norm = jnp.linalg.norm(r)
        theta = jnp.arcsin(R_earth/r_norm)

        # Find the actual angle between the position and the relative position
        dot_product = jnp.dot(-r/r_norm, rel_pos/jnp.linalg.norm(rel_pos))
        if dot_product > 1:
            dot_product = 0.9999
        elif dot_product < -1:
            dot_product = -0.9999

        angle = jnp.arccos(dot_product)

        # Check if the signal needs to pass through the Earth to detect the satellite
        if angle < theta:
            if r_norm*jnp.cos(angle) - jnp.sqrt(R_earth**2 - r_norm**2*jnp.sin(angle)**2) < jnp.linalg.norm(rel_pos):
                return (False, rel_pos, rel_vel)

        return (True, rel_pos, rel_vel)



    def get_measurement(self, satellite, t):

        (can_reach, rel_pos, rel_vel) = self.check_satellite_accessible(satellite, t)
        # if can_reach:
            # Return position measurement with measurement noise
        return rel_pos + np.random.multivariate_normal(np.zeros(3), self.Rr)

        # raise Exception("Always check measurability before attempting measurements!")



    def collect_measurements(self, t):

        self.y = jnp.zeros(self.y.size)
        r, v = self.get_state_eci(t)
        self.y = self.y.at[0:3].set(r)
        self.y = self.y.at[3:6].set(v)

        for i in range(len(self.rogues)):
            j = i + 2
            # reachable, _, _ = self.check_satellite_accessible(self.rogues[i], t)
            # if reachable:
            self.y = self.y.at[3*j:3*j+3].set(self.get_measurement(self.rogues[i], t))



    def measurement_model(self, mu):

        g = jnp.zeros(self.y.size)
        r_self, v_self = self.oe_to_eci(mu[0:6])
        g = g.at[0:3].set(r_self)
        g = g.at[3:6].set(v_self)

        for i in range(len(self.rogues)):
            r, _ = self.oe_to_eci(mu[6*i+6:6*i+12])
            g = g.at[3*i+6:3*i+9].set(r-r_self)

        return g



    def measurement_jacobian(self, mu):
        return jax.jacrev(self.measurement_model)(mu)



    def dynamics_model(self, mu):

        f = jnp.array(mu)
        for i in range(int(mu.size/6)):
            a = f[6*i]
            T = 2*jnp.pi*jnp.sqrt(a**3/398600)
            n = 2*jnp.pi/T
            f = f.at[6*i+5].set(mu[6*i+5] + n*self.dt)

        return f



    def dynamics_jacobian(self, mu):
        return jax.jacfwd(self.dynamics_model)(mu)
