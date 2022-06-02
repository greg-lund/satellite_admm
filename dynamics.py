
import numpy as np
import jax
import jax.numpy as jnp



def eci_to_oe(r, v):

    # Find the angular momentum. This is in the direction perpendicular to the perifocal frame (PQW).
    h = np.cross(r,v)
    h_norm = np.linalg.norm(h)
    W = h / h_norm

    # Find inclination and RAAN
    i = np.arctan2(np.sqrt(W[0]**2 + W[1]**2), W[2]) % (2*np.pi)
    Om = np.arctan2(W[0], -W[1]) % (2*np.pi)

    # Some parameters
    mu = 398600 # Earth gravitational parameter
    p = h_norm**2 / mu # Semi-latus rectum
    r_norm = np.linalg.norm(r) # Distance from the center of the Earth
    v_norm = np.linalg.norm(v) # Speed in ECI frame

    # Find semimajor axis, mean angular velocity, and eccentricity
    a = 1. / (2/r_norm - v_norm**2/mu)
    n = np.sqrt(mu / a**3)
    e = np.sqrt(1 - p/a)

    # Calculate eccentric, mean, and true anomaly
    E = np.arctan2(np.dot(r,v)/(a**2 * n), 1-r_norm/a) % (2*np.pi)
    M = (E - e*np.sin(E)) % (2*np.pi)
    nu = np.arctan2(np.sin(E) * np.sqrt(1-e**2), np.cos(E) - e) % (2*np.pi)

    # Calculate argument of periapsis
    u = np.arctan2(r[2]/np.sin(i), r[0]*np.cos(Om) + r[1]*np.sin(Om)) % (2*np.pi)
    w = (u - nu) % (2*np.pi)

    return a, e, i, Om, w, M



def eci_to_oe_mult(mu):
    oe = np.zeros(mu.size)
    for i in range(int(mu.size/6)):
        oe[6*i], oe[6*i+1], oe[6*i+2], oe[6*i+3], oe[6*i+4], oe[6*i+5] = eci_to_oe(mu[6*i:6*i+3], mu[6*i+3:6*i+6])
    return oe



def get_E(M, e):

    # Calculate the eccentric anomaly with the Newton-Rhapsod method
    E_prev = np.pi # Intiialization for previous iteration
    delta = 9999 # Initialization of delta (the change between each iteration)
    E = 0 # Initialization for next iteration
    eps = 10**-8 # Tolerance

    # Newton-Rhapsod method
    for counter in range(100):
        delta = -(E_prev - e*np.sin(E_prev) - M)/(1 - e*np.cos(E_prev))
        E = E_prev + delta
        E_prev = E

    return E



def oe_to_eci(oe):

    # Split the oe vector into the corresponding orbital elements
    a = oe[0]
    e = oe[1]
    i = oe[2]
    Om = oe[3]
    w = oe[4]
    M = oe[5]

    E = get_E(M, e) # Change the function to do this

    T = 2*np.pi*np.sqrt(a**3 / 398600)
    n = 2*np.pi/T

    # Obtain rotation matrices to rotate by RAAN, inclination, and argument of periapsis
    R1 = np.array([[np.cos(-Om), np.sin(-Om), 0], [-np.sin(-Om), np.cos(-Om), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, np.cos(-i), np.sin(-i)], [0, -np.sin(-i), np.cos(-i)]])
    R3 = np.array([[np.cos(-w), np.sin(-w), 0], [-np.sin(-w), np.cos(-w), 0], [0, 0, 1]])

    # Obtain positions and velocities in perifocal frame
    r_pqw = np.array([a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E), 0])
    v_pqw = a*n/(1-e*np.cos(E)) * np.array([-np.sin(E), np.sqrt(1-e**2)*np.cos(E), 0])

    # Apply rotation matrices to obtain r and v in the inertial frame
    r = R1 @ R2 @ R3 @ r_pqw
    v = R1 @ R2 @ R3 @ v_pqw

    return (r, v)



def propagate_eci_to_oe(mu, Sigma):

    # Sample particles, then propagate through nonlinear dynamics to get 95% confidence intervals
    N = 1000
    samples = np.zeros((N, mu.size))
    for i in range(int(mu.size/6.)):
        samples[:,6*i:6*i+6] = np.random.multivariate_normal(mu[6*i:6*i+6], Sigma[6*i:6*i+6,6*i:6*i+6], size=N)

    samples_oe = np.zeros(samples.shape)
    mu_oe = eci_to_oe_mult(mu) # Orbital element estimation

    # Get the orbital element samples from the particles
    bad_indices = []
    for i in range(N):
        samples_oe[i,:] = eci_to_oe_mult(samples[i,:])
        if np.isnan(samples_oe[i,:]).any():
            bad_indices.append(i)

    samples_oe_good = np.zeros((N-len(bad_indices), mu_oe.size))

    # Only keep the good samples
    counter = 0
    for i in range(N):
        if i in bad_indices:
            counter += 1
        else:
            samples_oe_good[i-counter,:] = samples_oe[i,:]


    mu_oe_dist = np.mean(samples_oe_good,axis=0) # Find the mean of the particles
    interval_oe = 2 * np.linalg.norm(samples_oe_good - mu_oe_dist, axis=0) / (N-len(bad_indices)-1) # 95% confidence interval

    return mu_oe, interval_oe



def gravity_sim(s):
    # s = [r, v]^T, concatenated among a bunch of satellites

    # Earth gravitational parameter
    mu = 398600.

    # Derivative of the position is simply the velocity in the inertial frame
    f = jnp.zeros(s.size)
    for i in range(int(s.size/6)):
        f = f.at[6*i+0].set(s[6*i+3])
        f = f.at[6*i+1].set(s[6*i+4])
        f = f.at[6*i+2].set(s[6*i+5])

        # Apply dv/dt = -mu/r^3 * vec(r)
        f = f.at[6*i+3].set(-mu/jnp.linalg.norm(s[6*i+0:6*i+3])**3 * s[6*i+0])
        f = f.at[6*i+4].set(-mu/jnp.linalg.norm(s[6*i+0:6*i+3])**3 * s[6*i+1])
        f = f.at[6*i+5].set(-mu/jnp.linalg.norm(s[6*i+0:6*i+3])**3 * s[6*i+2])

    return f



def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, dt=dt):
        k1 = dt * f(s)
        k2 = dt * f(s + k1 / 2)
        k3 = dt * f(s + k2 / 2)
        k4 = dt * f(s + k3)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator



def discrete_orbital_dynamics(f, dt):
    return discretize(f, dt)



def linearize_dynamics(fd, mu):
    return jax.jacfwd(fd)(mu)
