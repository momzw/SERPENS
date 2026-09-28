import multiprocessing
import warnings

import numpy as np

from ..parameters import GLOBAL_PARAMETERS
from ..serpens_simulation import SerpensSimulation
from .hotloop_bridge import advance_integrate_c

warnings.filterwarnings('ignore', category=RuntimeWarning, module='rebound')


class CerpensSimulation(SerpensSimulation):
    """
    Main class responsible for the Monte Carlo process of SERPENS.
    (Simulating the Evolution of Ring Particles Emergent from Natural Satellites)

    This class extends the REBOUND Simulation class to implement the SERPENS methodology
    for simulating particles emerging from natural satellites. It handles the creation,
    tracking, and evolution of particles in a gravitational system, with support for
    physical processes like thermal evaporation and sputtering.

    The simulation manages particle parameters through both REBOUNDx and HDF5 storage,
    allowing for efficient handling of large numbers of particles. It provides methods
    for advancing the simulation in time, adding particles, and configuring the
    simulation environment.

    Key features include:
    - Integration with REBOUND and REBOUNDx for accurate gravitational simulations
    - Support for multiple particle species with different physical properties
    - Efficient parameter storage using HDF5
    - Parallel particle generation using multiprocessing
    - Flexible time advancement options (by time, orbits, or spawning events)
    """

    supports_lorentz = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def advance_integrate(self, time):
        if not np.isfinite(time) or time <= 0:
            raise ValueError('Integration interval must be finite and positive')
        n_threads = GLOBAL_PARAMETERS.get("integration_threads", multiprocessing.cpu_count())
        target_time = self.t + time

        advance_integrate_c(
            sim=self,
            target_time=target_time,
            n_threads=n_threads,
            fix_circular=False,
            params=GLOBAL_PARAMETERS,
        )
