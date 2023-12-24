import rebound
import reboundx
import numpy as np
import warnings
import multiprocessing
import concurrent.futures
import pickle
from src.create_particle import create_particle
from src.parameters import Parameters
from tqdm import tqdm
import time


def weight_operator(sim_pointer, rebx_operator, dt):
    sim = sim_pointer.contents
    params = Parameters()
    for particle in sim.particles[sim.N_active:]:
        species_id = particle.params["serpens_species"]
        species = params.get_species(id=species_id)
        if species.duplicate is not None:
            species_id = int(str(species_id)[0])
            species = params.get_species(id=species_id)
        tau = species.network
        particle.params['serpens_weight'] *= np.exp(-sim.dt/tau)


def heartbeat(sim_pointer):
    """
    Not meant for external use.
    REBOUND heartbeat function to fix a source's orbit to be circular if the source is a moon.
    """
    sim = sim_pointer.contents

    if Parameters.int_spec["source_index"] > 2:
        for key, subdict in Parameters.celest.items():
            if "source" in subdict:
                par = subdict.copy()
        par.pop('primary', None)
        par.pop('source', None)

        p0 = rebound.Particle(simulation=sim, **par, primary=sim.particles["source_primary"])
        o = p0.orbit(primary=sim.particles["source_primary"], G=sim.G)
        mean_anomaly = o.n * sim.t
        p1 = rebound.Particle(simulation=sim, M=mean_anomaly, **par, primary=sim.particles["source_primary"])

        sim.particles["source"].xyz = p1.xyz
        sim.particles["source"].vxyz = p1.vxyz


class ReboundSetUp:
    def __init__(self, reb_sim):
        self.reb_sim = reb_sim

    def add_initial_particle(self, v_copy):
        source_is_planet = Parameters.int_spec["source_index"] == 2
        if source_is_planet:
            self.reb_sim.add(**v_copy, hash="source_primary")
        else:
            self.reb_sim.add(**v_copy, hash="star")

    def add_second_particle(self, v_copy):
        source_index = Parameters.int_spec["source_index"]
        source_is_planet = source_index == 2
        if source_is_planet:
            self.reb_sim.add(**v_copy, primary=self.reb_sim.particles[0], hash="source")
        else:
            self.reb_sim.add(**v_copy, primary=self.reb_sim.particles[0], hash="source_primary")

    def add_other_particles(self, v_copy):
        source_index = Parameters.int_spec["source_index"]
        source_is_planet = source_index == 2
        if not source_is_planet:
            if self.reb_sim.N == source_index - 1:
                self.reb_sim.add(**v_copy, primary=self.reb_sim.particles["source_primary"], hash='source')
            else:
                self.reb_sim.add(**v_copy, primary=self.reb_sim.particles["source_primary"])
        else:
            self.reb_sim.add(**v_copy, primary=self.reb_sim.particles["source_primary"])


def rebound_setup(params):
    """
     Not meant for external use.
     REBOUND simulation set up. Sets integrator and collision algorithm.
     Adds the gravitationally acting bodies to the simulation.
     Saves used parameters to drive.
     """
    print("Initializing new simulation instance...")

    reb_sim = rebound.Simulation()
    reb_sim.integrator = "whfast"  # Fast and unbiased symplectic Wisdom-Holman integrator.
    # reb_sim.ri_whfast.kernel = "lazy"
    reb_sim.collision = "direct"  # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    reb_sim.collision_resolve = "merge"
    if Parameters.int_spec["fix_source_circular_orbit"]:
        reb_sim.heartbeat = heartbeat

    # SI units:
    reb_sim.units = ('m', 's', 'kg')
    reb_sim.G = 6.6743e-11

    setup = ReboundSetUp(reb_sim)

    for k, v in Parameters.celest.items():
        if not type(v) == dict:
            continue
        else:
            v_copy = v.copy()
            v_copy.pop("primary", 0)
            v_copy.pop("source", 0)
            if reb_sim.N == 0:
                setup.add_initial_particle(v_copy)
            elif reb_sim.N == 1:
                setup.add_second_particle(v_copy)
            else:
                setup.add_other_particles(v_copy)

    reb_sim.N_active = len(Parameters.celest) - 1

    reb_sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    #
    # sim.ri_whfast.safe_mode = 0

    # REBOUNDX ADDITIONAL FORCES
    # ==========================
    rebx = reboundx.Extras(reb_sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = 3.e8
    reb_sim.particles["star"].params["radiation_source"] = 1
    rebx.register_param('serpens_species', 'REBX_TYPE_INT')
    rebx.register_param('serpens_weight', 'REBX_TYPE_DOUBLE')

    # Init save
    reb_sim.save_to_file("archive.bin", delete_file=True)
    rebx.save("rebx.bin")

    with open(f"Parameters.txt", "w") as f:
        f.write(f"{params.__str__()}")

    with open("Parameters.pkl", 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\t \t ... done!")

    return reb_sim, rebx


def create(source_state, source_r, phys_process, species):
    """
    Creates a batch of particles to be added to a SERPENS simulation.
    Utilizes multiprocessing and returns an array of particle state vectors.

    Arguments
    ---------
    source_state : array-like
        Coordinates and velocity components of the source to be added to created particles' states.
    source_r : float
        Radius of the source. Needed for correct anchoring of new particle vectors.
    phys_process : str
        Physical process responsible for the creation of particles.
        Currently implemented are thermal evaporation and sputtering.
    species : Species class instance
        Species to be created.
    """
    if phys_process == "thermal":
        n = species.n_th
    elif phys_process == "sputter":
        n = species.n_sp
    else:
        raise ValueError("Invalid process in particle creation.")

    if n == 0 or n is None:
        return np.array([])

    # Use the number of available CPU cores
    num_processes = min([multiprocessing.cpu_count(), n])

    def add_with_multiprocessing():
        per_create = int(n / num_processes)
        part_state = create_particle(species.id, process=phys_process, source=source_state, source_r=source_r,
                                     num=per_create)
        return part_state

    # Create a ThreadPoolExecutor with the desired number of threads/processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Submit function multiple times for parallel execution
        future_to_result = {executor.submit(add_with_multiprocessing): None for _ in range(num_processes)}

        # Collect the results from the futures
        results = []
        for future in concurrent.futures.as_completed(future_to_result):
            result = future.result()
            results.append(result)

    # Results as ndarray and reshape before returning
    results = np.asarray(results).reshape(np.shape(results)[0] * np.shape(results)[1], 6)

    return results


class SerpensSimulation:
    """
    Main class responsible for the Monte Carlo process of SERPENS.
    (Simulating the Evolution of Ring Particles Emergent from Natural Satellites)
    """

    def __init__(self, system='default'):
        """
        Initializes a REBOUND simulation instance, as well as used class instance variables.

        Arguments
        ---------
        system : str    (default: 'default')
            Name of the celestial system to be simulated.
            Valid are all names that have been set up in the src/objects.txt.
        """
        print("=======================================")
        print("SERPENS simulation has been created.")
        print("=======================================")

        self.params = Parameters()
        if not system == 'default':
            try:
                Parameters.modify_objects(celestial_name=system)
            except Exception:
                print("Cannot change the celestial objects. Are you sure you have implemented this system?")
                print("Exiting...")
                exit()

        # Create a new simulation
        self._sim, self._rebx = rebound_setup(self.params)

        self.var = {"iter": 0,
                    "source_a": self._sim.particles["source"].orbit(
                        primary=self._sim.particles["source_primary"]).a,
                    "source_P": self._sim.particles["source"].orbit(
                        primary=self._sim.particles["source_primary"]).P}

        self.var["boundary"] = self.params.int_spec["r_max"] * self.var["source_a"]

    def _add_particles(self):
        """
        Internal use only.
        Calls particle creation and adds the created particles to the REBOUND simulation instance.
        Saves particle hashes to a dictionary which contains information about a particle's species and weight factor.
        """
        source = self._sim.particles["source"]
        source_state = np.array([source.xyz, source.vxyz])

        if self.params.int_spec["gen_max"] is None or self.var['iter'] < self.params.int_spec["gen_max"]:
            for s in range(self.params.num_species):
                species = self.params.get_species(num=s + 1)

                rth = create(source_state, source.r, "thermal", species)
                rsp = create(source_state, source.r, "sputter", species)

                r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                for index, coord in enumerate(r):
                    identifier = f"{species.id}_{self.var['iter']}_{index}"
                    self._sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                  hash=identifier)

                    self._sim.particles[identifier].params["beta"] = species.beta
                    self._sim.particles[identifier].params["serpens_species"] = species.id
                    self._sim.particles[identifier].params["serpens_weight"] = 1.

    def integrate(self):
        weightop = self._rebx.create_operator("weightloss")
        weightop.operator_type = "recorder"
        weightop.step_function = weight_operator
        self._rebx.add_operator(weightop, dtfraction=1., timing="post")

        adv = self.var["source_P"] * self.params.int_spec["sim_advance"]
        self._sim.dt = adv / 10
        self._sim.integrate(adv * (self.var["iter"] + 1), exact_finish_time=0)

        # HAVE TO REMOVE BECAUSE OPERATOR CORRUPTS SAVE
        self._rebx.remove_operator(weightop)

    def single_advance(self):

        # ADD & REMOVE PARTICLES
        self._add_particles()

        self.integrate()

        remove = []
        for particle in self._sim.particles[self._sim.N_active:]:

            try:
                w = particle.params["serpens_weight"]
                species_id = particle.params["serpens_species"]
            except:
                print("Particle not found.")
                remove.append(particle.hash)
                continue

            species = self.params.get_species(id=species_id)
            mass_inject_per_advance = (species.mass_per_sec * self.params.int_spec["sim_advance"] *
                                       self.var["source_P"])
            pps = species.particles_per_superparticle(mass_inject_per_advance)

            particle_distance = np.linalg.norm(
                np.asarray(particle.xyz) - np.asarray(self._sim.particles["source_primary"].xyz))

            if particle_distance > self.var["boundary"] or w * pps < 1e10:
                try:
                    remove.append(particle.hash)
                except RuntimeError:
                    print("Removal error occurred.")
                    pass

        for hash in remove:
            self._sim.remove(hash=hash)

    def advance(self, num_sim_advances, save_freq=1, verbose=False):
        """
        Main function to be called for advancing the SERPENS simulation.
        Uses internal function to add particles, include loss for super-particles, and integrate in time using
        multiprocessing. Saves resulting REBOUND simulation state to disk.

        Arguments
        ---------
        num_sim_advances : int
            Number of advances to simulate.
        save_freq : int     (default: 1)
            Number of advances after which SERPENS saves the simulation instance.
        verbose : bool      (default: False)
            Enable printing of logs.
        """
        start_time = time.time()
        steady_state_counter = 0
        steady_state_breaker = None

        for _ in tqdm(range(num_sim_advances), disable=verbose):
            if verbose:
                print(f"Starting advance {self.var['iter']} ... ")

            n_before = self._sim.N
            self.single_advance()
            self._sim.save_to_file("archive.bin")
            self._rebx.save("rebx.bin")

            if verbose:
                t = self._sim.t
                print(f"Advance done! \n"
                      f"Simulation time [h]: {np.around(t / 3600, 2)} \n"
                      f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)} \n"
                      f"Number of particles: {self._sim.N}")

            # Handle steady state (1/2)
            if np.abs(self._sim.N - n_before) < 50:
                steady_state_counter += 1
                if steady_state_counter == 10 and self.params.int_spec["stop_at_steady_state"] is True:
                    print("Steady state reached!")
                    print("Stopping after another successful revolution...")
                    steady_state_breaker = 1
            else:
                steady_state_counter = 0

            # Handle steady state (2/2)
            if steady_state_breaker is not None:
                print(f"Advances left: {1 / self.params.int_spec['sim_advance'] - steady_state_breaker}")
                if steady_state_breaker == 1 / self.params.int_spec["sim_advance"]:
                    break
                else:
                    steady_state_breaker += 1

            # End iteration
            self.var["iter"] += 1
            if verbose:
                print("\t ... done!\n============================================")

        self.print_simulation_end_message()

    @staticmethod
    def print_simulation_end_message():
        with open('docs/sim_end_message.txt', 'r') as f:
            end_message = f.read()
        print(end_message)
