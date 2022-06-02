
from spacecraft import *
import jax
import jax.numpy as jnp
import numpy as np
from dynamics import *

class KnownSpacecraft(Spacecraft):

    def __init__(self, a, e, i, Om, w, M_start, rogues, dt=50):

        super().__init__(a, e, i, Om, w, M_start, dt)
        self.rogues = rogues
        self.y = np.zeros(3*len(rogues))
        self.R = np.diag(0.1 * np.ones(3*len(rogues)))
        self.rogue_indices_reachable = np.ones(len(rogues))
        self.Rr = np.eye(3) * 1
        self.Rv = np.eye(3) * 0.1
        self.r_belief = np.zeros(3)
        self.v_belief = np.zeros(3)



    def get_measurement(self, satellite, t):

        # Get the relative positition between the two satellites
        r_sat, v_sat = satellite.get_state_eci(t)
        r, v = self.get_state_eci(t)
        rel_pos = r_sat - r

        R_earth = 6371 # Radius of the Earth

        # Find the angle describing the cone that circumscribes the Earth
        r_norm = np.linalg.norm(r)
        theta = np.arcsin(R_earth/r_norm)

        # Find the actual angle between the position and the relative position
        dot_product = np.dot(-r/r_norm, rel_pos/jnp.linalg.norm(rel_pos))
        if dot_product > 1:
            dot_product = 0.9999
        elif dot_product < -1:
            dot_product = -0.9999

        angle = np.arccos(dot_product)

        # Check if the signal needs to pass through the Earth to detect the satellite
        if angle < theta:
            if r_norm*jnp.cos(angle) - np.sqrt(R_earth**2 - r_norm**2*np.sin(angle)**2) < np.linalg.norm(rel_pos):
                return False, rel_pos

        return True, rel_pos + np.random.multivariate_normal(np.zeros(3), self.Rr)



    def collect_measurements(self, t):
        self.rogue_indices_reachable = np.zeros(len(self.rogues))
        self.y = np.zeros(self.y.size)
        r, v = self.get_state_eci(t)
        self.r_belief = r + np.random.multivariate_normal(np.zeros(3), self.Rr)
        self.v_belief = v + np.random.multivariate_normal(np.zeros(3), self.Rv)

        for i in range(len(self.rogues)):
            can_reach, pos_est = self.get_measurement(self.rogues[i], t)
            if can_reach:
                self.y[3*i:3*i+3] = pos_est
                self.rogue_indices_reachable[i] = 1



    def measurement_model(self, mu):
        g = jnp.zeros(int((mu.size)/2))

        for i in range(int(mu.size/6)):
            r = mu[6*i:6*i+3]
            g = g.at[3*i:3*i+3].set(self.rogue_indices_reachable[i] * (r-self.r_belief))

        return g



    def measurement_jacobian(self, mu):
        return jax.jacfwd(self.measurement_model)(mu)
