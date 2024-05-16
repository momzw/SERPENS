import rebound
import reboundx
import numpy as np
import multiprocessing
import concurrent.futures
import pickle
import warnings
from src.create_particle import create_particle
from src.parameters import Parameters, NewParams
from tqdm import tqdm
import time

warnings.filterwarnings('ignore', category=RuntimeWarning, module='rebound')

def weight_operator(sim_pointer, rebx_operator, dt):
    sim = sim_pointer.contents
    params = Parameters()
    id_weight_multiplicator = {s.id: np.exp(-sim.dt/s.network) for _, s in params.species.items()}

    for particle in sim.particles[sim.N_active:]:
        particle.params['serpens_weight'] *= id_weight_multiplicator[particle.params["serpens_species"]]


def heartbeat(sim_pointer):
    """
    Not meant for external use.
    REBOUND heartbeat function to fix a source's orbit to be circular.
    """
    sim = sim_pointer.contents

    i = 0
    while True:
        try:
            primary = sim.particles[rebound.hash(sim.particles[f"source{i}"].params['source_primary'])]
            source = sim.particles[f"source{i}"]
            o = source.orbit(primary=primary)
            newP = rebound.Particle(simulation=sim, primary=primary, m=source.m, a=o.a, e=0, inc=o.inc,
                                    omega=o.omega, Omega=o.Omega, f=o.f)

            sim.particles[f"source{i}"].xyz = newP.xyz
            sim.particles[f"source{i}"].vxyz = newP.vxyz

            i += 1

        except rebound.ParticleNotFound:
            break


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


class SerpensSimulation(rebound.Simulation):
    """
    Main class responsible for the Monte Carlo process of SERPENS.
    (Simulating the Evolution of Ring Particles Emergent from Natural Satellites)
    """

    def __new__(cls, *args, **kwargs):
        return super(SerpensSimulation, cls).__new__(cls, *args, **kwargs)

    def __init__(self, system='default', init_serpens=True):
        """
        Initializes a REBOUND simulation instance, as well as used class instance variables.

        Arguments
        ---------
        system : str    (default: 'default')
            Name of the celestial system to be simulated.
            Valid are all names that have been set up in the src/objects.txt.
        init_serpens: bool (default: True)
            Initialize SERPENS. If False, base REBOUND simulation will be used.
        """
        super().__init__()
        self.params = Parameters()
        self.source_parameter_sets = []
        if system != 'default':
            try:
                Parameters.modify_objects(celestial_name=system)
            except Exception:
                print("Cannot load the celestial objects. Are you sure you have implemented this system?")
                print("Exiting...")
                exit()

        self.num_sources = 0
        self.serpens_iter = 0

        if init_serpens:
            self.rebound_setup()

    def rebound_setup(self):
        """
        Not meant for external use.
        REBOUND simulation set up. Sets integrator and collision algorithm.
        Adds the gravitationally acting bodies to the simulation.
        Saves used parameters to drive.
        """
        print("Initializing new simulation instance...")

        self.integrator = "whfast"  # Fast and unbiased symplectic Wisdom-Holman integrator.
        self.collision = "direct"  # Brute force collision search and scales as O(N^2).
        self.collision_resolve = "merge"
        if Parameters.int_spec["fix_source_circular_orbit"]:
            self.heartbeat = heartbeat

        # SI units:
        self.units = ('m', 's', 'kg')
        self.G = 6.6743e-11

        # REBOUNDx Additional Forces
        self.rebx = reboundx.Extras(self)
        rf = self.rebx.load_force("radiation_forces")
        self.rebx.add_force(rf)
        rf.params["c"] = 3.e8
        self.rebx.register_param('serpens_species', 'REBX_TYPE_INT')
        self.rebx.register_param('serpens_weight', 'REBX_TYPE_DOUBLE')
        self.rebx.register_param('source_primary', 'REBX_TYPE_INT')
        self.rebx.register_param('source_index', 'REBX_TYPE_INT')

        for k, v in Parameters.celest.items():
            if not type(v) == dict:
                continue
            else:
                v_copy = v.copy()
                if self.N == 0:
                    self.add(**v_copy, hash=k)
                    self.particles["star"].params["radiation_source"] = 1
                else:
                    self.add(**v_copy, hash=k)

        self.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

        # Init save
        self.save_to_file("archive.bin", delete_file=True)
        self.rebx.save("rebx.bin")

        with open(f"Parameters.txt", "w") as f:
            f.write(f"{self.params.__str__()}")

        with open("Parameters.pkl", 'wb') as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("\t \t ... done!")

        return self

    def add(self, particle=None, source=False, test_particle=False, **kwargs):

        if source:
            # First check if source has also an associated primary body.
            assert "primary" in kwargs, "Please provide the primary for a sourcing object."

            # Read whether the species should be modified.
            if "species" in kwargs:
                species = kwargs.pop("species")
                species = [species] if not isinstance(species, list) else species
                Parameters.modify_species(*species)

            # Add the source to the simulation.
            kwargs["hash"] = f"source{self.num_sources}"    # overwrites hash (if passed) for later reference.
            if particle is None:
                particle = rebound.Particle(simulation=self, **kwargs)
            super().add(particle)

            # Assign SERPENS parameters to the source particle.
            if isinstance(kwargs["primary"], rebound.Particle):
                self.particles[-1].params['source_primary'] = kwargs["primary"].hash.value
            elif isinstance(kwargs["primary"], str):
                self.particles[-1].params['source_primary'] = self.particles[kwargs["primary"]].hash.value
            else:
                raise TypeError(f"Unsupported type {type(kwargs['primary'])} for primary.")

            # Save the source parameters.
            self.source_parameter_sets.append(Parameters().get_current_parameters())
            with open('source_parameters.pkl', 'wb') as f:
                pickle.dump(self.source_parameter_sets, f)

            self.num_sources += 1
        else:
            if particle is None:
                particle = rebound.Particle(simulation=self, **kwargs)
            super().add(particle)

        if not test_particle:
            if self.N_active == -1:
                self.N_active += 2
            else:
                self.N_active += 1

    def _add_particles(self) -> None:
        """
        Internal use only.
        Calls particle creation and adds the created particles to the REBOUND simulation instance.
        Saves particle hashes to a dictionary which contains information about a particle's species and weight factor.
        """

        for source_index in range(self.num_sources):
            source = self.particles[f"source{source_index}"]
            source_state = np.array([source.xyz, source.vxyz])
            self._load_source_parameters(source_index)

            if self.params.int_spec["gen_max"] is None or self.serpens_iter < self.params.int_spec["gen_max"]:
                for s in range(self.params.num_species):
                    species = self.params.get_species(num=s + 1)

                    rth = create(source_state, source.r, "thermal", species)
                    rsp = create(source_state, source.r, "sputter", species)

                    r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                    for index, coord in enumerate(r):
                        identifier = f"{species.id}_{self.serpens_iter}_{source_index}_{index}"
                        self.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                 hash=identifier, test_particle=True)

                        self.particles[identifier].params["beta"] = species.beta
                        self.particles[identifier].params["serpens_species"] = species.id
                        self.particles[identifier].params["serpens_weight"] = 1.
                        self.particles[identifier].params["source_index"] = source_index

            Parameters.reset()

            # Need to make aware of all species for the integration weight operator:
            all_species = [s['species'][f'species{i+1}'] for s in self.source_parameter_sets for i in range(len(s['species']))]
            Parameters.modify_species(*all_species)

        self.rebx.save("rebx.bin")
        return

    def _load_source_parameters(self, source_index):
        parameter_set: dict = self.source_parameter_sets[source_index]
        NewParams(species=list(parameter_set['species'].values()),
                  int_spec=parameter_set['int_spec'],
                  therm_spec=parameter_set['therm_spec'],
                  celestial_name=parameter_set['celest']['SYSTEM-NAME']
                  )()

    def advance_integrate(self):

        primary = self.particles[rebound.hash(self.particles["source0"].params['source_primary'])]
        orbital_period0 = self.particles["source0"].orbit(primary=primary).P
        adv = orbital_period0 * self.params.int_spec["sim_advance"]
        self.dt = adv / 10

        threads_count = multiprocessing.cpu_count()

        particle_indices = list(range(self.N_active, self.N))
        particle_splits = np.array_split(particle_indices, threads_count)

        proc_indices = [list(range(self.N_active)) + test_split.tolist() for test_split in particle_splits]

        processes = []
        processes_rebx = []
        processes_operators = []
        for i in range(threads_count):
            copy = self.copy()

            copy.integrator = "whfast"
            copy.collision = "direct"
            copy.collision_resolve = "merge"
            if Parameters.int_spec["fix_source_circular_orbit"]:
                copy.heartbeat = heartbeat
            copy_rebx = reboundx.Extras(copy, "rebx.bin")

            weightop = copy_rebx.create_operator("weightloss")
            weightop.operator_type = "recorder"
            weightop.step_function = weight_operator
            copy_rebx.add_operator(weightop, dtfraction=1., timing="post")
            processes_operators.append(weightop)

            indices_to_keep_set = set(proc_indices[i])
            for j in reversed(range(copy.N)):
                if j not in indices_to_keep_set:
                    copy.remove(index=j)

            processes.append(copy)
            processes_rebx.append(copy_rebx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads_count) as executor:
            future_to_result = {
                executor.submit(
                    lambda x: x.integrate(adv * (self.serpens_iter + 1), exact_finish_time=0), p): p for p in processes
            }

            for future in concurrent.futures.as_completed(future_to_result):
                future.result()

        n_active = self.N_active
        del self.particles

        for i in range(n_active):
            self.add(processes[0].particles[i])

        for simulation, rebx in zip(processes, processes_rebx):
            for p in simulation.particles[simulation.N_active:]:
                self.add(p)
            rebx.detach(simulation)

        self.N_active = n_active
        self.t = processes[0].t

    def advance_single(self):

        # ADD & REMOVE PARTICLES
        self._add_particles()
        self.advance_integrate()

        primary = self.particles[rebound.hash(self.particles["source0"].params['source_primary'])]
        #orbital_period0 = self.particles["source0"].orbit(primary=primary).P
        boundary0 = self.params.int_spec["r_max"] * self.particles["source0"].orbit(primary=primary).a

        remove = []
        for particle in self.particles[self.N_active:]:

            #species = self.params.get_species(id=species_id)
            #mass_inject_per_advance = species.mass_per_sec * self.params.int_spec["sim_advance"] * orbital_period0
            #pps = species.particles_per_superparticle(mass_inject_per_advance)

            particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(primary.xyz))

            if particle_distance > boundary0: #or w * pps < 1e10:
                try:
                    remove.append(particle.hash)
                except RuntimeError:
                    print("Removal error occurred.")
                    pass

        for hash in remove:
            self.remove(hash=hash)

        self.save_to_file("archive.bin")
        self.rebx.save("rebx.bin")

    def advance(self, num_sim_advances, verbose=False):
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
                print(f"Starting SERPENS advance {self.serpens_iter} ... ")

            n_before = self.N
            self.advance_single()

            if verbose:
                t = self.t
                print(f"Advance done! \n"
                      f"Simulation time [h]: {np.around(t / 3600, 2)} \n"
                      f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)} \n"
                      f"Number of particles: {self.N}")

            # Handle steady state (1/2)
            if np.abs(self.N - n_before) < 50:
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
            self.serpens_iter += 1
            if verbose:
                print("\t ... done!\n============================================")

        self.print_simulation_end_message()

    @staticmethod
    def print_simulation_end_message():
        with open('docs/sim_end_message.txt', 'r') as f:
            end_message = f.read()
        print(end_message)
