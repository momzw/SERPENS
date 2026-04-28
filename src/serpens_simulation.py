import concurrent.futures
import copy
import multiprocessing
import os
import pickle
import time
import warnings

import h5py
import numpy as np
import rebound
import reboundx
from tqdm import tqdm

from src.parameters import GLOBAL_PARAMETERS
from src.spawner import generate_particles

warnings.filterwarnings('ignore', category=RuntimeWarning, module='rebound')


# Global h5 file access functions for use in callbacks
def get_particle_param_h5(particle_hash, param_name, h5_filename="simdata/particle_params.h5"):
    """
    Get a parameter value for a particle from the h5 file.
    This is a global function for use in callbacks like heartbeat.

    Arguments
    ---------
    particle_hash : str or int
        Hash of the particle
    param_name : str
        Name of the parameter to get
    h5_filename : str
        Name of the h5 file

    Returns
    -------
    The parameter value or None if not found
    """
    hash_str = str(particle_hash)
    try:
        with h5py.File(h5_filename, 'r') as f:
            if hash_str in f[param_name]:
                return f[param_name][hash_str][()]
    except (IOError, KeyError):
        pass
    return None


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


def get_species_by_name(name):
    for species in GLOBAL_PARAMETERS.get("all_species"):
        if species.name == name:
            return species
    return None


def get_species_by_id(id):
    for species in GLOBAL_PARAMETERS.get("all_species"):
        if species.id == id:
            return species
    return None


class SerpensSimulation(rebound.Simulation):
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

    def __new__(cls, *args, **kwargs):
        return super(SerpensSimulation, cls).__new__(cls, *args, **kwargs)

    def __init__(self, system=None, init_serpens=True):
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

        if system != None:
            try:
                GLOBAL_PARAMETERS.update_celest(celestial_name=system)
            except Exception:
                print("Cannot load the celestial objects. Are you sure you have implemented this system?")
                print("Exiting...")
                exit()

        self.source_parameter_sets = []
        self.num_sources = 0
        self.serpens_iter = 0
        self.source_obj_dict = {}
        self.obj_primary_dict = {}

        # Initialize h5 file for particle parameters
        self.init_h5_storage()

        if init_serpens:
            self.rebound_setup()

    def init_h5_storage(self):
        """
        Initialize h5 file for storing particle parameters.
        This replaces the REBOUNDx parameter storage for specific parameters.
        """
        self.h5_filename = "simdata/particle_params.h5"
        with h5py.File(self.h5_filename, 'w') as f:
            # Create groups for each parameter
            f.create_group("beta")
            f.create_group("serpens_species")
            f.create_group("source_hash")
            f.create_group("source_primary")
            f.create_group("serpens_creation_time")
            f.create_group("serpens_reaction_time")
            f.create_group("serpens_reaction_target")

    def get_particle_param(self, particle_hash, param_name):
        """
        Get a parameter value for a particle from the h5 file.

        Arguments
        ---------
        particle_hash : str or int
            Hash of the particle
        param_name : str
            Name of the parameter to get

        Returns
        -------
        The parameter value or None if not found
        """
        hash_str = str(particle_hash)
        with h5py.File(self.h5_filename, 'r') as f:
            if hash_str in f[param_name]:
                return f[param_name][hash_str][()]
        return None

    def set_particle_param(self, particle_hash, param_name, value):
        """
        Set a parameter value for a particle in the h5 file.

        Arguments
        ---------
        particle_hash : str or int
            Hash of the particle
        param_name : str
            Name of the parameter to set
        value : any
            Value to set
        """
        hash_str = str(particle_hash)
        with h5py.File(self.h5_filename, 'a') as f:
            if hash_str in f[param_name]:
                del f[param_name][hash_str]
            f[param_name].create_dataset(hash_str, data=value)

    def rebound_setup(self):
        """
        Not meant for external use.
        REBOUND simulation set up. Sets integrator and collision algorithm.
        Adds the gravitationally acting bodies to the simulation.
        Saves used parameters to drive.
        """
        print("Initializing new simulation instance...")

        self.integrator = "ias15"   # Integrator with Adaptive Step-size control, 15th order
        self.ri_ias15.min_dt = 1e-4
        self.collision = "direct"  # Brute force collision search and scales as O(N^2).
        self.collision_resolve = "merge"

        # SI units:
        self.units = ('m', 's', 'kg')
        self.G = 6.6743e-11

        # REBOUNDx Additional Forces
        self.rebx = reboundx.Extras(self)
        rf = self.rebx.load_force("radiation_forces")
        self.rebx.add_force(rf)
        rf.params["c"] = 3.e8

        # For CERPENS or catching errors in SERPENS
        self.rebx.register_param("q_over_m", "REBX_TYPE_DOUBLE")

        for k, v in GLOBAL_PARAMETERS.get('celest', {}).items():
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
        os.makedirs('simdata', exist_ok=True)
        self.save_to_file("simdata/archive.bin", delete_file=True)
        self.rebx.save("simdata/rebx.bin")

        with open("simdata/parameters.pkl", 'wb') as f:
            pickle.dump(GLOBAL_PARAMETERS.params, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("\t \t ... done!")

        return self

    def add(self, particle=None, test_particle=False, **kwargs):
        """
        Add a particle to the simulation.

        This method extends the REBOUND add method to track additional information
        about particles, particularly their primary bodies. It supports adding
        particles by direct specification or by name (for HORIZON database integration).

        Parameters
        ----------
        particle : rebound.Particle or str, optional
            Particle to add. If a string is provided, it's treated as a name for
            HORIZON database integration. If None, a particle is created from kwargs.
        test_particle : bool, default=False
            Whether to add the particle as a test particle.
        **kwargs : dict
            Additional keyword arguments for particle creation. Notable parameters:
            - hash : str
                Identifier for the particle
            - primary : rebound.Particle or str
                Primary body for the particle (for orbital elements)

        Returns
        -------
        None
        """
        if not isinstance(particle, str):
            if particle is None:
                particle = rebound.Particle(simulation=self, **kwargs)
            super().add(particle)

            if isinstance(kwargs.get('hash', None), str):
                object_hash = str(kwargs.get('hash'))
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

            # Load parameters specific to the current source
            parameter_set = self.source_parameter_sets[source_index]

            # Use GLOBAL_PARAMETERS as a context to temporarily modify
            with GLOBAL_PARAMETERS.as_context({
                'species': parameter_set['species'],
            }):
                # Generate particles only if maximum generation is not reached
                if GLOBAL_PARAMETERS.get("gen_max") is None or self.serpens_iter < GLOBAL_PARAMETERS.get("gen_max"):
                    for species in GLOBAL_PARAMETERS.get('species', []):
                        # Generate thermal and sputter particles
                        rth = create(source_state, source.r, "thermal", species)
                        rsp = create(source_state, source.r, "sputter", species)

                        # Combine results
                        r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                        # Add particles to the simulation
                        for index, coord in enumerate(r):
                            identifier = f"{species.id}_{self.serpens_iter}_{source_index}_{index}"
                            self.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                     hash=identifier, test_particle=True)

                            # Get the rebound hash of the added particle
                            particle_hash = self.particles[-1].hash.value

                            # Determine reaction at birth
                            reaction_time = self.t + np.random.exponential(scale=species.tau)   # Default: species lifetime
                            target_species_id = species.id  # Default: stays the same

                            if species.reactions:
                                # Sample exponential decay times for all possible reactions
                                pos = [coord[0], coord[1], coord[2]]
                                # TODO: Implement logic and not hardcode!
                                if any([react.lifetime_kwargs for react in species.reactions]):
                                    [react.lifetime_kwargs for react in species.reactions][0].update({'xyz_J': self.particles[1].xyz})

                                sampled_times = [np.random.exponential(r.get_lifetime(pos)) for r in species.reactions]
                                idx = np.argmin(sampled_times)

                                reaction_time = self.t + sampled_times[idx]
                                chosen_reaction = species.reactions[idx]

                                if chosen_reaction.target_species_name is None:
                                    target_species_id = -1  # Sentinel for removal
                                else:
                                    target_species_id = get_species_by_name(chosen_reaction.target_species_name).id

                            # Set particle-specific parameters in h5 file using the rebound hash
                            self.set_particle_param(particle_hash, "beta", species.beta)
                            self.set_particle_param(particle_hash, "serpens_species", species.id)
                            self.set_particle_param(particle_hash, "source_hash", source.hash.value)
                            self.set_particle_param(particle_hash, "serpens_creation_time", self.t)
                            self.set_particle_param(particle_hash, "serpens_reaction_time", reaction_time)
                            self.set_particle_param(particle_hash, "serpens_reaction_target", target_species_id)

                            # Set parameter for REBOUNDx
                            self.particles[identifier].params["beta"] = species.beta
                            self.particles[identifier].params["q_over_m"] = (
                                        species.q / species.m) if species.m != 0 else 0.0

        return

    def object_to_source(self, name, species):
        """
        Convert a celestial object to a particle source.

        This method designates an existing object in the simulation as a source
        for particle generation. It registers the object in the source dictionary,
        assigns it a source index, and sets up the necessary parameters for
        particle generation.

        Parameters
        ----------
        name : str
            Name (hash) of the object to convert to a source
        species : Species or list of Species
            Species of particles that this source will generate. Can be a single
            Species instance or a list of Species instances.

        Notes
        -----
        The method:
        1. Registers the object as a source with a unique index
        2. Sets the source's primary body using the obj_primary_dict
        3. Ensures all species are registered in GLOBAL_PARAMETERS
        4. Saves the source parameters to the source_parameters.pkl file

        The source will generate particles during simulation advancement based on
        the properties of the specified species.
        """
        self.source_obj_dict[f"source{self.num_sources}"] = name
        self.num_sources += 1

        # Get the rebound hash value of the particle
        particle_hash = self.particles[name].hash.value

        # Assign SERPENS parameters to the source particle using h5 storage.
        primary = self.obj_primary_dict[name]
        if isinstance(primary, rebound.Particle):
            primary_hash = primary.hash.value
        elif isinstance(primary, str):
            primary_hash = self.particles[primary].hash.value
        else:
            raise TypeError(f"Unsupported type {type(primary)} for primary.")

        self.set_particle_param(particle_hash, 'source_primary', primary_hash)

        species: list = [species] if not isinstance(species, list) else species
        # Ensure all Species variants are uniquely identifiable *within this source*.
        # If multiple instances of the same chemical species are provided without an explicit
        # `duplicate=...`, they would otherwise share the same `id` and get merged in analysis.
        used_ids: set[int] = {s.id for s in GLOBAL_PARAMETERS.get('all_species', [])}
        for s in species:
            # Backwards compatibility for older pickled Species objects.
            if not hasattr(s, "type_id"):
                s.type_id = s.id
            if not hasattr(s, "duplicate"):
                s.duplicate = None

            # Normalize variant id for auto-dup logic.
            if s.duplicate is None:
                s.id = s.type_id

            if s.id in used_ids:
                if s.duplicate is not None:
                    raise ValueError(
                        "Duplicate Species instance id collision detected. "
                        "At least two Species variants in the same `species=[...]` list evaluate to the same `id`, "
                        "but one of them already has an explicit `duplicate` value."
                    )
                dup_idx = 1
                while (s.type_id * 10 + dup_idx) in used_ids:
                    dup_idx += 1
                s.duplicate = dup_idx
                s.id = s.type_id * 10 + dup_idx

            used_ids.add(s.id)

        species_registered = [s.id in [rs.id for rs in GLOBAL_PARAMETERS.get('all_species', [])] for s in species]
        for i, s in enumerate(species_registered):
            if not s:
                all_species = GLOBAL_PARAMETERS.get('all_species', [])
                all_species.append(species[i])
                GLOBAL_PARAMETERS.set('all_species', all_species)
                with open("simdata/parameters.pkl", 'wb') as f:
                    pickle.dump(GLOBAL_PARAMETERS.params, f, protocol=pickle.HIGHEST_PROTOCOL)

        with GLOBAL_PARAMETERS.as_context(dict(species=species)):
            # Save the source parameters.
            self.source_parameter_sets.append(
                copy.deepcopy(
                    GLOBAL_PARAMETERS.params
                )
            )
            with open('simdata/source_parameters.pkl', 'wb') as f:
                pickle.dump(self.source_parameter_sets, f, protocol=pickle.HIGHEST_PROTOCOL)

    def advance_integrate(self, time):

        if GLOBAL_PARAMETERS.get("lorentz_enabled", True):
            raise Exception(
                "Lorentz force is only supported in CERPENS (C-version). \n" +
                'Please set GLOBAL_PARAMETERS.set("lorentz_enabled", False)'
            )

        threads_count = multiprocessing.cpu_count()
    
        particle_indices = list(range(self.N_active, self.N))
        particle_splits = np.array_split(particle_indices, threads_count)
    
        proc_indices = [list(range(self.N_active)) + test_split.tolist() for test_split in particle_splits]
    
        processes = []
        processes_rebx = []
        for i in range(threads_count):
            copy = self.copy()
    
            copy.integrator = "ias15"
            copy.ri_ias15.min_dt = 1e-4
            copy.collision = "direct"
            copy.collision_resolve = "merge"
            copy_rebx = reboundx.Extras(copy, "simdata/rebx.bin")

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

    def _fix_source_orbits_circular(self):
        """Fix all source orbits to be circular (e=0).

        Called once per SERPENS step — negligible cost compared to integration,
        and avoids the issues that arise from modifying positions mid-integration
        (heartbeat or REBOUNDx operator) which can trigger spurious collisions.
        """
        for src_idx in range(self.num_sources):
            src_name = self.source_obj_dict[f"source{src_idx}"]
            source = self.particles[src_name]
            primary_hash = self.get_particle_param(source.hash.value, 'source_primary')
            if primary_hash is None:
                continue
            primary = self.particles[rebound.hash(int(primary_hash))]
            o = source.orbit(primary=primary)
            newP = rebound.Particle(simulation=self, primary=primary, m=source.m,
                                    a=o.a, e=0, inc=o.inc, omega=o.omega,
                                    Omega=o.Omega, f=o.f)
            source.xyz = newP.xyz
            source.vxyz = newP.vxyz

    def advance_single(self, time=None, orbit_object=None):
        """
        Advance the simulation by a single step.

        This method performs a single integration step, adding new particles,
        advancing the simulation by the specified time, and removing particles
        that have moved beyond the boundary.

        Parameters
        ----------
        time : float, optional
            Time to advance the simulation (in seconds)
        orbit_object : str or rebound.Particle, optional
            Object to use as reference for calculating the boundary radius

        Notes
        -----
        Particles are removed if they move beyond a distance of r_max * semi-major axis
        from the primary body, where r_max is defined in GLOBAL_PARAMETERS.

        The simulation state is saved to disk after the integration step.
        """
        # ADD & REMOVE PARTICLES
        self._add_particles()
        self.rebx.save("simdata/rebx.bin")
        self.advance_integrate(time=time)

        # Fix source orbits to circular after integration (before boundary checks)
        if GLOBAL_PARAMETERS.get('fix_source_circular_orbit', True):
            self._fix_source_orbits_circular()

        # Get the rebound hash value of the orbit_object
        orbit_object_hash = self.particles[orbit_object].hash.value

        # Get source_primary from h5 file
        source_primary_hash = self.get_particle_param(orbit_object_hash, 'source_primary')
        if source_primary_hash is None:
            try:
                # Fall back to REBOUNDx if not found in h5
                source_primary_hash = self.particles[orbit_object].params.get('source_primary')
            except AttributeError:
                # Fall back to object zero
                pass

        if source_primary_hash is not None:
            primary = self.particles[rebound.hash(int(source_primary_hash))]
        else:
            primary = self.particles[0]

        boundary0 = GLOBAL_PARAMETERS.get("r_max", 10) * self.particles[orbit_object].orbit(primary=primary).a

        remove = []
        reacting = 0
        for particle in self.particles[self.N_active:]:
            particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(primary.xyz))

            if particle_distance > boundary0:
                try:
                    remove.append(particle.hash)
                except RuntimeError:
                    print("Removal error occurred.")
                    pass
                finally:
                    continue
            # Reaction logic
            rx_time = self.get_particle_param(particle.hash.value, "serpens_reaction_time")
            if rx_time <= self.t:
                target_id = self.get_particle_param(particle.hash.value, "serpens_reaction_target")

                if target_id == -1 or target_id is None:
                    remove.append(particle.hash)
                else:
                    # Perform the conversion
                    new_species = get_species_by_id(target_id)
                    new_reaction_time = self.t + np.random.exponential(scale=new_species.tau)
                    self.set_particle_param(particle.hash.value, "serpens_species", target_id)
                    self.set_particle_param(particle.hash.value, "serpens_reaction_time", new_reaction_time)
                    self.set_particle_param(particle.hash.value, "serpens_creation_time", self.t)
                    particle.params["q_over_m"] = new_species.q / new_species.m
                    reacting += 1

                    # TODO: Need to set other parameters from birth (new reaction targets etc.)

        print(f"Reacting particles: {reacting}")
        print(f"Removing {len(remove)} particles.")
        for particle_hash in remove:
            self.remove(hash=particle_hash)

        self.save_to_file("simdata/archive.bin")
        self.rebx.save("simdata/rebx.bin")
        # No need to explicitly save the h5 file as it's saved on each parameter update

    def advance(self, hours=None, days=None, orbits=None, orbits_reference=None, spawns=None, verbose=False, dt_scale_factor=1):
        """
        Main function to be called for advancing the SERPENS simulation.

        This method advances the simulation by a specified amount of time, which can be
        specified in hours, days, or orbital periods. It handles particle creation,
        integration, and removal, and saves the simulation state to disk after each step.

        The simulation can be advanced in a single step or in multiple smaller steps
        (specified by the spawns parameter), which allows for particle creation at
        regular intervals during the simulation.

        Parameters
        ----------
        hours : float, optional
            Number of hours to advance the simulation
        days : float, optional
            Number of days to advance the simulation
        orbits : float, optional
            Number of orbital periods to advance the simulation
        orbits_reference : str or rebound.Particle, optional
            Object to use as reference for calculating orbital periods.
            If None, "source0" is used.
        spawns : int, optional
            Number of separate integration steps to use. If provided, the total
            simulation time is divided into this many equal steps, with particle
            creation occurring at each step.
        verbose : bool, default=False
            Whether to print detailed progress information

        Notes
        -----
        At least one of hours, days, or orbits must be provided to specify the
        simulation duration. If multiple are provided, they are added together.

        The method saves the simulation state to disk after each integration step,
        including the REBOUND archive, REBOUNDx data, and HDF5 parameter storage.
        """
        if orbits_reference is None:
            orbit_reference_object = 1
        else:
            orbit_reference_object = orbits_reference

        # Get the rebound hash value of the orbit_reference_object
        orbit_reference_object_hash = self.particles[orbit_reference_object].hash.value

        # Get source_primary from h5 file
        source_primary_hash = self.get_particle_param(orbit_reference_object_hash, 'source_primary')
        if source_primary_hash is None:
            try:
                # Fall back to REBOUNDx if not found in h5
                source_primary_hash = self.particles[orbit_reference_object].params.get('source_primary')
            except AttributeError:
                # Fall back to object zero
                pass

        if source_primary_hash is not None:
            primary = self.particles[rebound.hash(int(source_primary_hash))]
        else:
            primary = self.particles[0]

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
