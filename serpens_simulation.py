import rebound
import reboundx
import numpy as np
import multiprocessing
import concurrent.futures
import pickle
import warnings
from src.spawner import generate_particles
from src.parameters import Parameters, NewParams
from tqdm import tqdm
import time

warnings.filterwarnings('ignore', category=RuntimeWarning, module='rebound')


def heartbeat(sim_pointer):
    """
    TODO: Fix the hashing
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
        part_state = generate_particles(
            species.id, process=phys_process, source=source_state, source_r=source_r,
            n_samples=per_create
        )

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
    results += source_state.reshape(6)

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
        self.source_obj_dict = {}
        self.obj_primary_dict = {}

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
        self.rebx.register_param('source_primary', 'REBX_TYPE_INT')
        self.rebx.register_param('source_hash', 'REBX_TYPE_INT')

        for k, v in Parameters.celest.items():
            if not type(v) == dict:
                continue
            else:
                v_copy = v.copy()
                if self.N == 0:
                    self.add(**v_copy, hash=k)
                    self.particles[0].params["radiation_source"] = 1
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

    def add(self, particle=None, test_particle=False, **kwargs):
        if not isinstance(particle, str):
            if particle is None:
                particle = rebound.Particle(simulation=self, **kwargs)
            super().add(particle)

            if isinstance(kwargs.get('hash', None), str):
                object_hash = kwargs['hash']
                primary = kwargs.get('primary', None)
                self.obj_primary_dict[object_hash] = primary
                self.particles[-1].hash = object_hash

        # Preparing HORIZON database implementation
        else:
            primary = kwargs.get('primary', None)
            super().add(particle, **kwargs)
            self.obj_primary_dict[particle] = primary
            #self.particles[-1].hash = particle

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
            source_str = self.source_obj_dict[f"source{source_index}"]
            source = self.particles[source_str]
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
                        self.particles[identifier].params["source_hash"] = source.hash.value

            Parameters.reset()

            # Need to make aware of all species for the integration weight operator:
            all_species = [s['species'][f'species{i+1}'] for s in self.source_parameter_sets for i in range(len(s['species']))]
            Parameters.modify_species(*all_species)

        self.rebx.save("rebx.bin")
        return

    def object_to_source(self, name, species):
        self.source_obj_dict[f"source{self.num_sources}"] = name
        self.num_sources += 1

        # Assign SERPENS parameters to the source particle.
        primary = self.obj_primary_dict[name]
        if isinstance(primary, rebound.Particle):
            self.particles[name].params['source_primary'] = primary.hash.value
        elif isinstance(primary, str):
            self.particles[name].params['source_primary'] = self.particles[primary].hash.value
        else:
            raise TypeError(f"Unsupported type {type(primary)} for primary.")

        species = [species] if not isinstance(species, list) else species
        Parameters.modify_species(*species)
        # Save the source parameters.
        self.source_parameter_sets.append(Parameters().get_current_parameters())
        with open('source_parameters.pkl', 'wb') as f:
            pickle.dump(self.source_parameter_sets, f)

    def _load_source_parameters(self, source_index):
        parameter_set: dict = self.source_parameter_sets[source_index]
        NewParams(species=list(parameter_set['species'].values()),
                  int_spec=parameter_set['int_spec'],
                  therm_spec=parameter_set['therm_spec'],
                  celestial_name=parameter_set['celest']['SYSTEM-NAME']
                  )()

    def advance_integrate(self, time):
        threads_count = multiprocessing.cpu_count()

        particle_indices = list(range(self.N_active, self.N))
        particle_splits = np.array_split(particle_indices, threads_count)

        proc_indices = [list(range(self.N_active)) + test_split.tolist() for test_split in particle_splits]

        processes = []
        processes_rebx = []
        for i in range(threads_count):
            copy = self.copy()

            copy.integrator = "whfast"
            copy.collision = "direct"
            copy.collision_resolve = "merge"
            if Parameters.int_spec["fix_source_circular_orbit"]:
                copy.heartbeat = heartbeat
            copy_rebx = reboundx.Extras(copy, "rebx.bin")

            indices_to_keep_set = set(proc_indices[i])
            for j in reversed(range(copy.N)):
                if j not in indices_to_keep_set:
                    copy.remove(index=j)

            processes.append(copy)
            processes_rebx.append(copy_rebx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads_count) as executor:
            future_to_result = {
                executor.submit(
                    lambda x: x.integrate(time * (self.serpens_iter + 1), exact_finish_time=0), p): p for p in processes
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

    def advance_single(self, time=None, orbit_object=None):
        # ADD & REMOVE PARTICLES
        self._add_particles()
        self.advance_integrate(time=time)

        primary = self.particles[rebound.hash(self.particles[orbit_object].params['source_primary'])]
        boundary0 = self.params.int_spec["r_max"] * self.particles[orbit_object].orbit(primary=primary).a

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

    def advance(self, hours=None, days=None, orbits=None, orbits_reference=None, spawns=None, verbose=False):
        """
        Main function to be called for advancing the SERPENS simulation.
        Uses internal function to add particles, include loss for super-particles, and integrate in time using
        multiprocessing. Saves resulting REBOUND simulation state to disk.
        """
        if orbits_reference is None:
            orbit_reference_object = self.source_obj_dict["source0"]
        else:
            orbit_reference_object = orbits_reference
        primary = self.particles[rebound.hash(self.particles[orbit_reference_object].params['source_primary'])]
        orbital_period = self.particles[orbit_reference_object].orbit(primary=primary).P

        total_time = 0
        if days is not None:
            total_time += days * 3600 * 24
        if hours is not None:
            total_time += hours * 3600
        if orbits is not None:
            total_time += orbits * orbital_period

        if spawns is not None:
            iteration_length = total_time / spawns
            iterations = spawns
        else:
            iteration_length = total_time
            iterations = 1
        self.dt = iteration_length / 12     # Fixed value.

        start_time = time.time()
        for _ in tqdm(range(iterations), disable=verbose):
            if verbose:
                print(f"Starting SERPENS integration step {self.serpens_iter} ... ")

            self.advance_single(iteration_length, orbit_reference_object)

            if verbose:
                t = self.t
                print(
                    f"Step done! \n"
                    f"Simulation time [h]: {np.around(t / 3600, 2)} \n"
                    f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)} \n"
                    f"Number of particles: {self.N} \n"
                )

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
