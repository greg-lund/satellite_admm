
import jax
import jax.numpy as jnp
import numpy as np
from dynamics import *

class Spacecraft:

    def __init__(self, a, e, i, Om, w, M_start, dt=50):

        self.a = a # Semimajor axis (km)
        self.e = e # Eccentricity
        self.i = i # Inclination (rad)
        self.w = w # Argument of the periapsis (rad)
        self.Om = Om # RAAN (rad)
        self.M_start = M_start # Starting mean anomaly (rad)
        self.M = M_start

        self.T = 2*np.pi*np.sqrt(self.a**3/398600) # Orbital period (s)
        self.n = 2*np.pi/self.T

        self.Qr = np.eye(3) * 5
        self.Qv = np.eye(3) * 0.5

        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.t = -1.
        self.dt = dt



    def get_state_eci(self, t):

        if self.t != t:
            self.set_state_eci(t)

        return (self.r, self.v)



    def set_state_eci(self, t):

        self.M = (self.n*t + self.M_start) % (2*np.pi) # Current mean anomaly
        r, v = oe_to_eci([self.a, self.e, self.i, self.Om, self.w, self.M])

        self.r = r
        self.v = v
        self.t = t



    def apply_control(self, dv, t):

        r, v = self.get_state_eci(t)
        self.v = v + dv
        self.a, self.e, self.i, self.Om, self.w, self.M = eci_to_oe(self.r, self.v)
