
from spacecraft import *
from dynamics import *
import numpy as np
import jax
import jax.numpy as jnp

class Constellation:

    def __init__(known_satellites, rogue_satellites, sim_time):
        self.known_satellites = known_satellites
        self.rogue_satellites = rogue_sattelites
        self.T = sim_time
        self.t = np.linspace(0, self.T, 1)
