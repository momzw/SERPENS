import numpy as np
import os as os
import glob
import shutil
import rebound
import reboundx
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from ctypes import c_uint

from datetime import datetime
from src import DTFE, DTFE3D
from src.parameters import GLOBAL_PARAMETERS
from src.visualize import Visualize


mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{wasysym}')


def ensure_data_loaded(method):
    """
    Decorator for the SerpensAnalyzer class.
    Ensures the correct data is loaded for a given timestep before method execution.
    """

    def wrapper(self, timestep, *args, **kwargs):
        self.load_timestep_data(timestep=timestep)
        return method(self, timestep, *args, **kwargs)

    return wrapper


class SerpensAnalyzer:
    """
    SERPENS analyzer class that calculates particle densities from SERPENS simulation data.

    This class uses Delaunay Field Density Estimations to analyze particle distributions
    and provides methods for visualizing the results. It loads simulation data from archive files,
    processes particle positions and properties, applies various filters and transformations,
    and generates visualizations of the particle distributions.

    The analyzer supports various types of visualizations including planar views, line-of-sight
    projections, 3D plots, and 1D density cuts. It can also calculate and plot phase curves
    for different species.
    """

    # Class constants
    OUTPUT_DIR = 'output'
    ARCHIVE_FILENAME = 'simdata/archive.bin'
    PARAMS_FILENAME = 'simdata/parameters.pkl'
    SOURCE_PARAMS_FILENAME = 'simdata/source_parameters.pkl'

    def __init__(self, save_output=False, save_archive=False, folder_name=None,
                 z_cutoff=None, r_cutoff=None, v_cutoff=None, reference_system=None):
        """
        Initialize the analyzer by loading the archive.bin and hash dictionary files.
        We state whether outputs shall be saved or not, including the archive files.
        The analyzer allows for removal of particles before density calculations depending on certain criteria.
        Currently implemented criteria are distance from orbital plane (z_cutoff),
        radial distance from primary (r_cutoff), or velocity cutoff (v_cutoff).

        Arguments
        ---------
        save_output : bool      (default: False)
            Save plots after creation.

        save_archive : bool     (default: False)
            Save archive.bin and hash dictionary to folder_name.
        folder_name : str       (default: None -> UTC of execution)
            Name of folder (relative to run path) to save files to. If not provided, folders will be created in the
            'output' sub-folder, named by UTC of execution and source type.
        z_cutoff : float        (default: None)
            Vertical cutoff distance above/below source orbital plane in units of source primary radius.
            Particles beyond this vertical distance will not be considered in the analysis.
        r_cutoff : float        (default: None)
            Radial cutoff distance to source primary in units of source primary radius.
            Particles beyond this radial distance will not be considered in the analysis.
        v_cutoff : float        (default: None)
            Velocity cutoff of particles in units of meter per second. This is an upper limit.
            Particles with velocities greater than v_cutoff will not be considered in the analysis.
        """

        #try:
        #    with open('Parameters.pkl', 'rb') as handle:
        #        params_load = pickle.load(handle)
        #        params_load()
        #except Exception:
        #    raise Exception("Parameters.pkl not found.")

        self._load_parameters()
        self.sa = self._load_simulation_archive()

        # Configuration settings
        self.save_output = save_output
        self.save_archive = save_archive
        self.save_index = 1
        self.reference_system = reference_system

        # Data storage
        self.sim = None
        self.particle_positions = None
        self.particle_velocities = None
        self.particle_hashes = None
        self.particle_species = None
        self.particle_weights = None
        self.cached_timestep = None
        self.rotated_timesteps = []

        # Cutoff parameters
        self.cutoffs = {"z": z_cutoff, "r": r_cutoff, "v": v_cutoff}

        # Output directory setup
        if save_output:
            self._setup_output_directory(folder_name)

    def _setup_output_directory(self, folder_name):
        """
        Set up output directory for saving results.

        Creates the necessary directory structure for saving output files and plots.
        If requested, also copies the simulation archive to the output directory.

        Parameters
        ----------
        folder_name : str or None
            Name of the folder to save results to. If None, a folder named with the current
            date and time will be created in the OUTPUT_DIR.
        """
        print("Setting up output directory...")

        # Create directory path
        self.path = folder_name if folder_name else datetime.now().strftime("%d%m%Y--%H-%M-%S")
        output_path = f'{self.OUTPUT_DIR}/{self.path}'
        plots_path = f'{output_path}/plots'

        # Create directories if needed
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # Copy archive if requested
        if self.save_archive:
            print("\tCopying simulation archive...")
            archive_path = f"{os.getcwd()}/{self.ARCHIVE_FILENAME}"
            destination = f"{os.getcwd()}/{output_path}"
            shutil.copy2(archive_path, destination)

    def _load_parameters(self):
        """
        Load global and source-specific parameters from files.

        Attempts to load global parameters from PARAMS_FILENAME and source-specific 
        parameters from SOURCE_PARAMS_FILENAME. If the files don't exist or can't be 
        loaded, default values (empty dictionary or list) are used instead.

        The loaded parameters are stored in GLOBAL_PARAMETERS.params and 
        self.source_parameter_sets respectively.
        """
        # Load global parameters
        if os.path.exists(self.PARAMS_FILENAME):
            try:
                with open(self.PARAMS_FILENAME, 'rb') as f:
                    GLOBAL_PARAMETERS.params = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading global parameters: {e}")
                GLOBAL_PARAMETERS.params = {}
        else:
            print("Global parameters file not found. Using default parameters.")
            GLOBAL_PARAMETERS.params = {}

        # Load source-specific parameters
        if os.path.exists(self.SOURCE_PARAMS_FILENAME):
            try:
                with open(self.SOURCE_PARAMS_FILENAME, 'rb') as f:
                    self.source_parameter_sets = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading source-specific parameters: {e}")
                self.source_parameter_sets = []
        else:
            print("Source-specific parameters file not found. Using an empty list.")
            self.source_parameter_sets = []

    def get_particle_param(self, particle_hash, param_name, h5_filename="simdata/particle_params.h5"):
        """
        Get a parameter value for a particle from the h5 file.
        Falls back to REBOUNDx if not found in h5.

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

        # Fall back to REBOUNDx if not found in h5
        try:
            return self.sim.particles[particle_hash].params[param_name]
        except (AttributeError, KeyError):
            return None

    @staticmethod
    def _load_simulation_archive():
        """Loads simulation archive.bin file."""
        try:
            return rebound.Simulationarchive("simdata/archive.bin", process_warnings=True)
        except Exception:
            raise FileNotFoundError("Simulation archive not found.")

    def get_primary(self, source_hash) -> rebound.Particle:
        """
        Get the primary body for a given source.

        Parameters
        ----------
        source_hash : str or None
            Hash of the source or None for system primary

        Returns
        -------
        rebound.Particle
            Primary particle object
        """
        if source_hash is not None:
            # Get source_primary from h5 file with fallback to REBOUNDx
            if isinstance(source_hash, str):
                source_hash_value = self.sim.particles[source_hash].hash.value
            else:
                source_hash_value = source_hash.value
            source_primary_hash = self.get_particle_param(source_hash_value, 'source_primary')
            return self.sim.particles[rebound.hash(int(source_primary_hash))]
        else:
            return self.sim.particles[0]

    def load_timestep_data(self, timestep=None, time=None):
        """
        Load particle data for a given timestep.

        This method serializes particle vectors and attributes and
        sets the REBOUND simulation instance to the specified timestep.

        Parameters
        ----------
        timestep : int
            Simulation timestep to load
        """
        if timestep is not None:
            if self.cached_timestep == timestep:
                return

            self.cached_timestep = timestep
            self.sim = self.sa[int(timestep)]
            _ = reboundx.Extras(self.sim, "simdata/rebx.bin")
        elif time is not None:
            self.sim = self.sa.getSimulation(time)
            _ = reboundx.Extras(self.sim, "simdata/rebx.bin")
        else:
            raise ValueError("timestep or time must be specified.")

        if self.reference_system is not None and timestep not in self.rotated_timesteps:
            self._rotate_reference_system()
            self.rotated_timesteps.append(timestep)

        self.particle_positions = np.zeros((self.sim.N, 3), dtype="float64")
        self.particle_velocities = np.zeros((self.sim.N, 3), dtype="float64")
        self.particle_hashes = np.zeros(self.sim.N, dtype="uint32")
        self.sim.serialize_particle_data(
            xyz=self.particle_positions,
            vxvyvz=self.particle_velocities,
            hash=self.particle_hashes
        )

        self.particle_species = np.zeros(self.sim.N, dtype="int")
        self.particle_weights = np.zeros(self.sim.N, dtype="float64")
        # Store source hashes as strings to match h5 storage format
        self._particle_source_hashes = np.zeros(self.sim.N, dtype=object)

        for k1 in range(self.sim.N):
            try:
                part_hash = int(self.particle_hashes[k1])
                # Get parameters from h5 file with fallback to REBOUNDx
                part_creation_time = self.get_particle_param(part_hash, 'serpens_creation_time')
                species_id = self.get_particle_param(part_hash, 'serpens_species')
                source_hash = self.get_particle_param(part_hash, 'source_hash')

                if part_creation_time is None or species_id is None or source_hash is None:
                    continue

                species = [s for s in GLOBAL_PARAMETERS.get('all_species') if s.id == species_id][0]
                self.particle_species[k1] = species_id
                self.particle_weights[k1] = np.exp(
                    -(self.sim.t - part_creation_time) / species.network
                )
                self._particle_source_hashes[k1] = source_hash
            except (AttributeError, IndexError):
                continue

        self.source_hashes = []
        # Error correction:
        for h in np.unique(self._particle_source_hashes):
            if h == 0:  # Skip default values
                continue
            try:
                _ = self.sim.particles[rebound.hash(int(h))]
                self.source_hashes.append(rebound.hash(int(h)))
            except (rebound.ParticleNotFound, ValueError):
                continue

        self.num_sources = len(self.source_hashes)
        self._apply_masks()

    def _calculate_offsets(self, plane):
        """
        Calculate coordinate offsets.

        Parameters
        ----------
        plane : str
            Plane for calculating offsets ('xy', 'yz', or '3d')

        Returns
        -------
        tuple
            Coordinate offsets (x, y, z)
        """

        primary = self.get_primary(self.reference_system)
        if plane == 'xy':
            return primary.x, primary.y, 0
        elif plane == 'yz':
            return primary.y, primary.z, 0
        elif plane == '3d':
            return primary.x, primary.y, primary.z
        else:
            raise ValueError("Invalid plane.")

    def _rotate_reference_system(self):
        """
        Apply coordinate transformation to particles if using geocentric reference.

        Rotates all particles based on the primary's position and inclination.
        """
        primary = self.get_primary(self.reference_system)
        phase = np.arctan2(primary.y, primary.x)

        try:
            inc = primary.orbit().inc
        except ValueError:
            inc = 0

        reb_rot = rebound.Rotation(angle=phase, axis='z')
        reb_rot_inc = rebound.Rotation(angle=inc, axis='y')
        for particle in self.sim.particles:
            particle.rotate(reb_rot.inverse())
            particle.rotate(reb_rot_inc)

    def _apply_masks(self):
        """
        Apply filters to particles based on cutoff parameters.

        Internal use only. This method filters particles based on the cutoff parameters
        specified during initialization (z_cutoff, r_cutoff, and v_cutoff). For each source,
        it creates a mask that identifies particles to keep based on:

        - z_cutoff: Vertical distance from the source's orbital plane
        - r_cutoff: Radial distance from the source's primary
        - v_cutoff: Velocity relative to the source

        The method updates the particle data arrays to include only particles that pass
        all the specified filters. This is done separately for each source, and the results
        are combined to create the final filtered dataset.
        """
        masks_for_each_source = []

        for source_hash in self.source_hashes:
            source_primary = self.get_primary(source_hash)

            # Compare as strings to handle both string and int storage
            source_particles_mask = np.array([str(h) == str(source_hash.value) for h in self._particle_source_hashes])

            combined_mask = np.ones(len(self.particle_positions[source_particles_mask]), dtype=bool)

            for cutoff_type in self.cutoffs:
                cutoff_value = self.cutoffs[cutoff_type]

                if cutoff_value is not None:
                    assert isinstance(cutoff_value, (float, int))
                    condition = None

                    if cutoff_type == "z":
                        condition = (self.particle_positions[source_particles_mask, 2] < cutoff_value * source_primary.r) & \
                                    (self.particle_positions[source_particles_mask, 2] > -cutoff_value * source_primary.r)
                    elif cutoff_type == "r":
                        r = np.linalg.norm(self.particle_positions[source_particles_mask] - source_primary.xyz, axis=1)
                        condition = r < cutoff_value * source_primary.r
                    elif cutoff_type == "v":
                        v = np.linalg.norm(self.particle_velocities[source_particles_mask] - self.sim.particles[source_hash].vxyz, axis=1)
                        condition = v < cutoff_value

                    if condition is not None:
                        combined_mask &= condition

            # Keep the indices to preserve for this source
            indices_to_keep_source = np.where(source_particles_mask)[0][combined_mask]

            # Create a full size mask for this source's kept indices (initially all False)
            full_size_mask = np.zeros(len(self.particle_positions), dtype=bool)

            # Mark the indices to keep as True
            full_size_mask[indices_to_keep_source] = True

            # Save this source mask
            masks_for_each_source.append(full_size_mask)

        # Overall mask (use logical OR operation over the masks from all sources)
        overall_mask = np.logical_or.reduce(masks_for_each_source)

        # Then apply the overall_mask to your arrays:
        self.particle_positions = self.particle_positions[overall_mask]
        self.particle_velocities = self.particle_velocities[overall_mask]
        self.particle_hashes = self.particle_hashes[overall_mask]
        self.particle_species = self.particle_species[overall_mask]
        self.particle_weights = self.particle_weights[overall_mask]

    @ensure_data_loaded
    def delaunay_field_estimation(self, timestep: int, species, d=2, los=False):
        """
        Calculate particle density values using Delaunay Triangulation Field Estimation.

        This is the main function for density estimation, which initializes the DTFE estimator
        at a specified timestep. It automatically loads the necessary data through the
        ensure_data_loaded decorator. The method restricts analysis to a single particle
        species and can perform either 2D or 3D density estimation.

        The method calculates both the spatial distribution of particles and their physical
        weights based on the simulation parameters and particle lifetimes.

        Parameters
        ----------
        timestep : int
            Simulation timestep at which to calculate densities.
        species : Species class instance
            Particle species to be analyzed.
        d : int, default=2
            Dimension of analysis:
            - For d=2: Returns densities in particles per cm²
            - For d=3: Returns densities in particles per cm³
        los : bool, default=False
            Line-of-sight mode:
            - If True: Views system from positive x-axis and masks particles hidden
              behind specified objects
            - If False: No masking is applied (suitable for orbital plane views)

        Returns
        -------
        tuple
            A tuple containing:
            - points: Array of particle positions
            - dens: Array of density values at each point
            - phys_weights: Array of physical weights for each particle
        """
        points_mask = np.where(self.particle_species == species.id)

        points = self.particle_positions[points_mask]
        velocities = self.particle_velocities[points_mask]
        weights = self.particle_weights[points_mask]
        source_hashes = self._particle_source_hashes[points_mask]

        # Physical weight calculation:
        total_injected = timestep * (species.n_sp + species.n_th)
        remaining_part = len(points[:, 0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * self.sim.t
        number_of_particles = mass_in_system / species.m
        phys_weights = number_of_particles * weights/np.sum(weights)

        if d == 2:

            if los:
                masks_for_each_source = []

                # Determine which objects to use for masking
                masking_objects = []

                # Default: use all objects with radius > 0
                for i in range(self.sim.N):
                    particle = self.sim.particles[i]
                    if hasattr(particle, 'r') and particle.r > 0:
                        masking_objects.append(i)
                    else:
                        break

                # Process each source
                for source_hash in self.source_hashes:
                    source_primary = self.get_primary(source_hash)
                    # Handle comparison regardless of whether source_hash is stored as string or int
                    if isinstance(source_hash, c_uint):
                        source_hash_value = source_hash.value
                    else:
                        source_hash_value = source_hash

                    # Compare as strings to handle both string and int storage
                    source_particles_mask = np.array([str(h) == str(source_hash_value) for h in source_hashes])

                    # Create a mask for particles to be masked (density set to 0)
                    particles_to_mask = np.zeros(np.sum(source_particles_mask), dtype=bool)

                    # Apply masking for each object
                    for obj_idx in masking_objects:
                        obj = self.sim.particles[obj_idx]
                        if hasattr(obj, 'r') and obj.r > 0:
                            los_dist_to_obj = np.sqrt((points[source_particles_mask, 1] - obj.y) ** 2 +
                                                     (points[source_particles_mask, 2] - obj.z) ** 2)

                            # Identify particles behind the object
                            behind_obj_mask = ((los_dist_to_obj < obj.r) &
                                              (points[source_particles_mask, 0] - obj.x < 0))

                            particles_to_mask |= behind_obj_mask

                    # Create a full-size mask for the particles to be masked
                    full_size_mask = np.zeros(len(points), dtype=bool)
                    indices_to_mask = np.where(source_particles_mask)[0][particles_to_mask]
                    full_size_mask[indices_to_mask] = True
                    masks_for_each_source.append(full_size_mask)

                dtfe = DTFE.DTFE(points[:, 1:3], velocities[:, 1:3], phys_weights)

                dens = dtfe.density(points[:, 1], points[:, 2]) / 1e4
                dens[np.logical_or.reduce(masks_for_each_source)] = 0
            else:
                dtfe = DTFE.DTFE(points[:, :2], velocities[:, :2], phys_weights)
                dens = dtfe.density(points[:, 0], points[:, 1]) / 1e4

        elif d == 3:
            dtfe = DTFE3D.DTFE(points, velocities, phys_weights)
            dens = dtfe.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6

        else:
            raise ValueError("Invalid dimension in DTFE.")

        dens[dens < 0] = 0

        return dens, dtfe.delaunay

    @ensure_data_loaded
    def get_statevectors(self, timestep):
        """
        Returns all particle positions and velocities as a tuple of array-likes at a given timestep.
        Automatically runs the "pull_data" function through the decorator.

        Arguments
        ---------
        timestep : int
            Passed to data pull in order to ensure that the correct state vectors are returned.
        """
        return self.particle_positions, self.particle_velocities

    def _create_visualizer(self, perspective='planar', **kwargs):
        """
        Create a visualizer object with appropriate configuration.

        Parameters
        ----------
        perspective : str
            Visualization perspective ('planar' or 'los')
        **kwargs : dict
            Additional visualization parameters

        Returns
        -------
        Visualize
            Configured visualizer object
        """
        reference = self.reference_system
        if reference is None:
            reference = self.sim.N_active - 1

        vis_perspective = 'los' if perspective == 'los' else 'planar'
        return Visualize(self.sim, reference, perspective=vis_perspective, **kwargs)

    def _save_visualization(self, vis, timestep, prefix, show=True):
        """
        Save visualization output and handle potential saving bugs.

        Parameters
        ----------
        vis : Visualize
            Visualizer object
        timestep : int
            Current timestep
        prefix : str
            Filename prefix
        show : bool
            Whether to display the visualization
        """
        if self.save_output:
            filename = f'{prefix}_{timestep}_{self.save_index}'
            vis(show_bool=show, save_path=self.path, filename=filename)
            self.save_index += 1

            # Handle saving bugs
            plot_path = f'./output/{self.path}/plots'
            list_of_files = glob.glob(f'{plot_path}/*')
            if not list_of_files:
                return True  # No files to check

            latest_file = max(list_of_files, key=os.path.getctime)
            if os.path.getsize(latest_file) < 50000:
                print(f"\tDetected low filesize (threshold at {50000 / 1000} KB). "
                      "Possibly encountered a saving bug. Retrying process.")
                os.remove(latest_file)
                return False  # Save failed
            return True  # Save succeeded
        else:
            vis(show_bool=show)
            return True  # No save needed

    def _process_species(self, vis, timestep, d=3, scatter=True, triplot=False, perspective='planar', **kwargs):
        """
        Process and visualize all particle species.

        Parameters
        ----------
        vis : Visualize
            Visualizer object
        timestep : int
            Current timestep
        d : int
            Dimension of density calculation
        scatter : bool
            Whether to create a scatter plot
        triplot : bool
            Whether to plot the Delaunay tessellation
        perspective : str
            Perspective for visualization ('planar' or 'los')
        **kwargs : dict
            Additional visualization parameters
        """
        all_species = GLOBAL_PARAMETERS.get('all_species', [])
        num_species = len(all_species)

        for k in range(num_species):
            species = all_species[k]
            is_los = perspective == 'los'

            # Get positions for this species
            mask = self.particle_species == species.id
            points = self.particle_positions[mask]

            if len(points) == 0:
                vis.empty(k)
                continue

            # Perform density calculation
            dens, delaunay = self.delaunay_field_estimation(
                timestep, species, d=(2 if is_los else 3), los=is_los,
            )

            if scatter:
                if is_los:
                    # Handle line-of-sight specific rendering
                    self._add_los_scatter(vis, points, dens, k, **kwargs)
                else:
                    # Handle planar scatter
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, d=d)

            if triplot and not is_los:
                if d == 3:
                    vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices[:, :3])
                elif d == 2:
                    vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices)

        # Set title for planar view
        if perspective == 'planar' and num_species > 0:
            vis.set_title(fr"Particle Densities $log_{{10}} (N/\mathrm{{cm}}^{{{-d}}})$", size=25, color='w')

    def _add_los_scatter(self, vis, points, dens, species_index, **kwargs):
        """Add line of sight scatter plot with planet masking."""
        # Auto-adjust visualization levels if specified
        if kwargs.get('lvlmin') == 'auto':
            vis.vis_params.update({"lvlmin": np.log10(np.min(dens[dens > 0])) - 0.5})
        if kwargs.get('lvlmax') == 'auto':
            vis.vis_params.update({"lvlmax": np.log10(np.max(dens[dens > 0])) + 0.5})

        # Apply planet masking
        primary = self.get_primary(self.reference_system)
        los_dist_to_planet = np.sqrt((points[:, 1] - primary.y) ** 2 +
                                     (points[:, 2] - primary.z) ** 2)
        mask = (los_dist_to_planet > primary.r) | \
               (points[:, 0] - np.abs(primary.x) > 0)

        # Add scatter for visible particles
        vis.add_densityscatter(
            species_index,
            -points[:, 1][mask],
            points[:, 2][mask],
            dens[mask],
            d=2,
            zorder=10
        )

    def plot_planar(self, timestep, d=3, scatter=True, triplot=False, show=True, **kwargs):
        """
        Plot the system from a top-down perspective (orbital plane).

        Parameters
        ----------
        timestep : int or array-like
            Timestep(s) at which to create the plot
        d : int
            Dimension of density calculation (2=particles/cm², 3=particles/cm³)
        scatter : bool
            Whether to create a scatter plot with density coloring
        triplot : bool
            Whether to plot the Delaunay tessellation
        show : bool
            Whether to show the plot
        **kwargs : dict
            Additional visualization parameters
        """
        ts_list = np.atleast_1d(timestep).astype(int)

        for ts in ts_list:
            self.load_timestep_data(timestep=ts)
            vis = self._create_visualizer(perspective='planar', **kwargs)

            self._process_species(
                vis, ts, d=d, scatter=scatter, triplot=triplot,
                perspective='planar', **kwargs
            )

            self._save_visualization(vis, ts, 'TD', show=show)
            del vis

    def plot_lineofsight(self, timestep, show=True, scatter=True, **kwargs):
        """
        Plot the system from a line-of-sight perspective.

        Parameters
        ----------
        timestep : int or array-like
            Timestep(s) at which to create the plot
        show : bool
            Whether to show the plot
        scatter : bool
            Whether to create a scatter plot with density coloring
        **kwargs : dict
            Additional visualization parameters
        """
        ts_list = np.atleast_1d(timestep).astype(int).tolist()

        running_index = 0
        while running_index < len(ts_list):
            ts = ts_list[::-1][running_index]
            self.load_timestep_data(timestep=ts)

            vis = self._create_visualizer(perspective='los', **kwargs)

            self._process_species(
                vis, ts, d=2, scatter=scatter, perspective='los', **kwargs
            )

            save_success = self._save_visualization(vis, ts, 'LOS', show=show)
            if save_success:
                running_index += 1

            del vis

    @ensure_data_loaded
    def plot_3d(self, timestep, species_num=1, log_cutoff=None, show_star=False):
        """
        Uses the plotly module to create an interactive 3D plot.

        Arguments
        ---------
        timestep : int
            Timestep at which to create the plot.
        species_num : int   (default: 1)
            The number/index of the species. If more than one species is present, set '2', '3', ... to access the
            corresponding species.
        log_cutoff : float      (default: None)
            Include a log(density) cutoff. log in base 10.
        """
        species = GLOBAL_PARAMETERS.get('all_species', [])[species_num - 1]
        pos = self.particle_positions[np.where(self.particle_species == species.id)]
        dens, _ = self.delaunay_field_estimation(timestep, species, d=3)

        phi, theta = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
        x_p = (self.get_primary(self.reference_system).r * np.sin(theta) * np.cos(phi) +
               self.get_primary(self.reference_system).x)
        y_p = (self.get_primary(self.reference_system).r * np.sin(theta) * np.sin(phi) +
               self.get_primary(self.reference_system).y)
        z_p = (self.get_primary(self.reference_system).r * np.cos(theta) +
               self.get_primary(self.reference_system).z)

        np.seterr(divide='ignore')

        if log_cutoff is not None:
            df = pd.DataFrame({
                'x': pos[:, 0][np.log10(dens) > log_cutoff],
                'y': pos[:, 1][np.log10(dens) > log_cutoff],
                'z': pos[:, 2][np.log10(dens) > log_cutoff]
            })
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.log10(dens[np.log10(dens) > log_cutoff]), opacity=.3,
                                color_continuous_scale='YlOrBr')
        else:
            df = pd.DataFrame({
                'x': pos[:, 0],
                'y': pos[:, 1],
                'z': pos[:, 2]
            })
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.log10(dens), opacity=.3)

        fig.add_trace(go.Surface(x=x_p, y=y_p, z=z_p, surfacecolor=np.zeros(shape=x_p.shape), showscale=False,
                                 colorscale='matter'))
        if show_star:
            x_s = (self.sim.particles["star"].r * np.sin(theta) * np.cos(phi) +
                   self.sim.particles["star"].x)
            y_s = (self.sim.particles["star"].r * np.sin(theta) * np.sin(phi) +
                   self.sim.particles["star"].y)
            z_s = (self.sim.particles["star"].r * np.cos(theta) +
                   self.sim.particles["star"].z)
            fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, surfacecolor=np.zeros(shape=x_p.shape), showscale=False,
                                     colorscale='Hot'))

        fig.update_coloraxes(colorbar_exponentformat='e')
        fig.update_layout(scene_aspectmode='data')

        np.seterr(divide='warn')
        return fig

    @ensure_data_loaded
    def plot_1d_cut(self, timestep, species_num=1, log_scale=False, max_distance_rp=6, **kwargs):
        species = GLOBAL_PARAMETERS.get('all_species', [])[species_num - 1]
        points_mask = np.where(self.particle_species == species.id)

        points = self.particle_positions[points_mask]
        velocities = self.particle_velocities[points_mask]
        weights = self.particle_weights[points_mask]
        source_hashes = self._particle_source_hashes[points_mask]

        # Physical weight calculation:
        total_injected = timestep * (species.n_sp + species.n_th)
        remaining_part = len(points[:, 0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * self.sim.t
        number_of_particles = mass_in_system / species.m
        phys_weights = number_of_particles * weights / np.sum(weights)

        masks_for_each_source = []
        for source_hash in self.source_hashes:
            source_primary = self.get_primary(source_hash)
            # Handle comparison regardless of whether source_hash is stored as string or int
            if isinstance(source_hash, c_uint):
                source_hash_value = source_hash.value
            else:
                source_hash_value = source_hash

            # Compare as strings to handle both string and int storage
            source_particles_mask = np.array([str(h) == str(source_hash_value) for h in source_hashes])

            los_dist_to_planet = np.sqrt((points[source_particles_mask, 1] - source_primary.y) ** 2 +
                                         (points[source_particles_mask, 2] - source_primary.z) ** 2)

            mask = ((los_dist_to_planet < source_primary.r) &
                    (points[source_particles_mask, 0] - source_primary.x < 0))

            indices_to_keep = np.where(source_particles_mask)[0][mask]
            full_size_mask = np.zeros(len(points), dtype=bool)
            full_size_mask[indices_to_keep] = True
            masks_for_each_source.append(full_size_mask)

        dtfe = DTFE.DTFE(points[:, 1:3], velocities[:, 1:3], phys_weights)

        dens = dtfe.density(points[:, 1], points[:, 2]) / 1e4
        dens[np.logical_or.reduce(masks_for_each_source)] = 0

        # 1D CUT
        source = self.sim.particles[self.source_hashes[0]]
        x_linspace = np.linspace(source.y + source.r, source.y + max_distance_rp*source.r, 100)
        y_fixed = np.full(100, source.z)
        dens_cut = dtfe.density(x_linspace, y_fixed) / 1e4
        dens_cut_smooth = gaussian_filter1d(dens_cut, sigma=3)

        plt.figure(dpi=200, **kwargs)
        plt.plot((x_linspace - source.y) / source.r, dens_cut)
        plt.plot((x_linspace - source.y) / source.r, dens_cut_smooth, c='r')
        if log_scale:
            plt.yscale("log")
        plt.xlabel(r"$R_P$")
        plt.ylabel(r"$N\ [\mathrm{cm}^{-2}]$")
        plt.tight_layout()
        plt.show()

    def calculate_phasecurve(self, source_name, species_name=None, orbits=1):
        last_sim = self.sa[-1]
        _ = reboundx.Extras(last_sim, "simdata/rebx.bin")

        source_particle = last_sim.particles[source_name]
        # Get source_primary from h5 file with fallback to REBOUNDx
        source_primary_hash = self.get_particle_param(source_particle.hash.value , 'source_primary')
        source_primary = last_sim.particles[rebound.hash(int(source_primary_hash))]
        source_orbit = source_particle.orbit(primary=source_primary)

        simulation_step_timedelta = self.sa[2].t - self.sa[1].t
        orbit_start_time = last_sim.t - orbits*source_orbit.P

        timesteps = np.arange(orbit_start_time / simulation_step_timedelta, len(self.sa))

        if species_name is None:
            species = GLOBAL_PARAMETERS.get('all_species')[0]
        else:
            assert isinstance(species_name, str)
            species = [s for s in GLOBAL_PARAMETERS.get('all_species') if s.name == species_name][0]

        phase_data = []
        for t in tqdm(timesteps, desc="Calculating phase curve"):
            self.load_timestep_data(timestep=t)
            dens2d, _ = self.delaunay_field_estimation(t, species, d=2, los=True)
            dens3d, _ = self.delaunay_field_estimation(t, species, d=3)

            log_dens2d = np.log10(dens2d + 1e-5)
            log_dens3d = np.log10(dens3d + 1e-5)
            log_dens2d[log_dens2d < 0] = 0
            log_dens3d[log_dens3d < 0] = 0

            part = self.sim.particles[source_name]
            # Get source_primary from h5 file with fallback to REBOUNDx
            source_primary_hash = self.get_particle_param(part.hash.value, 'source_primary')
            primary = self.sim.particles[rebound.hash(int(source_primary_hash))]
            phase = part.orbit(primary=primary).theta * 180 / np.pi

            # if self.reference_system == 'geocentric':
            #    shift = (timestep * self.params.int_spec["sim_advance"] * exomoon_orbit.P / exoplanet_orbit.P) * 360
            #    phases[i] = (phase + shift) % 360
            # else:
            #    phases[i] = phase
            # if phases[-1] == phases[0]:
            #    phases[-1] += 360

            phase_data.append([
                phase, t, np.max(log_dens2d), np.mean(log_dens2d), np.max(log_dens3d), np.mean(log_dens3d)
            ])

        df = pd.DataFrame(data=phase_data, columns=['Phase', 'Timestep', 'Max_2D', 'Mean_2D', 'Max_3D', 'Mean_3D'])
        df.to_csv('phase-curve.csv', index=False)

    @staticmethod
    def plot_phasecurve(filename='phase-curve.csv', column_density=True, particle_density=True, type="max"):
        df_phase = pd.read_csv(filename).sort_values(by='Phase')
        #df_phase.loc[df_phase.index[-1], "Phase"] = 360

        # Number of subplots
        rows = column_density + particle_density

        # Create subplot layout
        #fig = px.scatter()
        fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.1)

        phases = df_phase['Phase'].values
        col = 'Max' if type == "max" else "Mean"

        x_ = np.linspace(min(phases), max(phases), 200)

        row_idx = 1
        if particle_density:
            dens3d_ = df_phase[f'{col}_3D'].values
            # Smooth the curve and plot
            spl = make_interp_spline(phases, dens3d_, k=3)
            y_ = spl(x_)

            # Add particle density trace
            fig.add_trace(
                go.Scatter(
                    x=x_,
                    y=y_,
                    mode='lines',
                    line=dict(width=3),
                    opacity=0.9,
                    name='Particle Density'
                ),
                row=row_idx, col=1
            )

            # Update y-axis title for this subplot
            fig.update_yaxes(
                title_text=r"$\mathrm{log}\ \bar{n}\ [\mathrm{cm}^{-3}]$",
                title_font=dict(size=18),
                tickfont=dict(size=19),
                row=row_idx, col=1
            )
            row_idx += 1

        if column_density:
            dens2d_ = df_phase[f'{col}_2D'].values
            spl = make_interp_spline(phases, dens2d_, k=3)

            y_ = spl(x_)
            # Add column density trace
            fig.add_trace(
                go.Scatter(
                    x=x_,
                    y=y_,
                    mode='lines',
                    line=dict(width=3),
                    opacity=0.9,
                    name='Column Density'
                ),
                row=row_idx, col=1
            )

            # Update y-axis title for this subplot
            fig.update_yaxes(
                title_text=r"$\mathrm{log}\ \bar{N}\ [\mathrm{cm}^{-2}]$",
                title_font=dict(size=20),
                tickfont=dict(size=22),
                row=row_idx, col=1
            )

            # Only add x-axis label to the last subplot
            fig.update_xaxes(
                title_text=r"exomoon phase $\phi$",
                title_font=dict(size=20),
                tickfont=dict(size=22),
                range=[0, 360],
                row=row_idx, col=1
            )

        # Set common x-axis properties for all subplots
        fig.update_xaxes(range=[0, 360])
        fig.update_layout(
            width=900,
            height=450 * rows,
            showlegend=True
        )
        fig.show(block=True)
