
import jax
import jax.numpy as jnp

class Spacecraft:

    def __init__(self, a, e, i, Om, w, M_start):

        self.a = a # Semimajor axis (km)
        self.e = e # Eccentricity
        self.i = i # Inclination (rad)
        self.w = w # Argument of the periapsis (rad)
        self.Om = Om # RAAN (rad)
        self.M_start = M_start # Starting mean anomaly (rad)
        self.M = M_start

        self.T = 2*jnp.pi*jnp.sqrt(self.a**3/398600) # Orbital period (s)
        self.n = 2*jnp.pi/self.T

        self.Qr = jnp.eye(3) * 0.1
        self.Qv = jnp.eye(3) * 0.005

        self.Rr = jnp.eye(3) * 0.1
        self.Rv = jnp.eye(3) * 0.1

        self.r = jnp.zeros(3)
        self.v = jnp.zeros(3)
        self.t = -1.



    def get_true_anomaly(self, E):
        # Find true anomaly from eccentric anomaly
        nu = jnp.arctan2(jnp.sin(E)*jnp.sqrt(1 - self.e**2), jnp.cos(E) - self.e)
        return nu



    def get_E(self, M, e):

        # Calculate the eccentric anomaly with the Newton-Rhapsod method
        E_prev = jnp.pi # Intiialization for previous iteration
        delta = 9999 # Initialization of delta (the change between each iteration)
        E = 0 # Initialization for next iteration
        eps = 10**-8 # Tolerance

        # Newton-Rhapsod method
        for counter in range(100):
            delta = -(E_prev - e*jnp.sin(E_prev) - M)/(1 - e*jnp.cos(E_prev))
            E = E_prev + delta
            E_prev = E

        return E



    def get_state_eci(self, t):

        if self.t != t:
            self.set_state_eci(t)

        return (self.r, self.v)



    def set_state_eci(self, t):

        self.M = (self.n*t + self.M_start) % (2*jnp.pi) # Current mean anomaly
        r, v = self.oe_to_eci(jnp.array([self.a, self.e, self.i, self.Om, self.w, self.M]))

        # Add process noise
        # r = r + jnp.random.multivariate_normal(jnp.zeros(3), self.Qr)
        # v = v + jnp.random.multivariate_normal(jnp.zeros(3), self.Qv)

        self.r = r
        self.v = v
        self.t = t



    def oe_to_eci(self, oe):

        a = oe[0]
        e = oe[1]
        i = oe[2]
        Om = oe[3]
        w = oe[4]
        M = oe[5]

        E = self.get_E(M, e) # Change the function to do this

        T = 2*jnp.pi*jnp.sqrt(a**3/398600)
        n = 2*jnp.pi/T

        # Obtain rotation matrices to rotate by RAAN, inclination, and argument of periapsis
        R1 = jnp.array([[jnp.cos(-Om), jnp.sin(-Om), 0], [-jnp.sin(-Om), jnp.cos(-Om), 0], [0, 0, 1]])
        R2 = jnp.array([[1, 0, 0], [0, jnp.cos(-i), jnp.sin(-i)], [0, -jnp.sin(-i), jnp.cos(-i)]])
        R3 = jnp.array([[jnp.cos(-w), jnp.sin(-w), 0], [-jnp.sin(-w), jnp.cos(-w), 0], [0, 0, 1]])

        # Obtain positions and velocities in perifocal frame
        r_pqw = jnp.array([a*(jnp.cos(E)-e), a*jnp.sqrt(1-e**2)*jnp.sin(E), 0])
        v_pqw = a*n/(1-e*jnp.cos(E)) * jnp.array([-jnp.sin(E), jnp.sqrt(1-e**2)*jnp.cos(E), 0])

        # Apply rotation matrices to obtain r and v in the inertial frame
        r = R1 @ R2 @ R3 @ r_pqw
        v = R1 @ R2 @ R3 @ v_pqw

        return (r, v)
