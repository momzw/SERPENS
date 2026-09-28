import glob
import os as os
import pickle
import shutil
from ctypes import c_uint
from datetime import datetime

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rebound
import reboundx
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import QhullError
from tqdm import tqdm

from src import DTFE, DTFE3D
from src import grain_analysis
from src.grains import GRAIN_METADATA_FIELDS
from src.parameters import GLOBAL_PARAMETERS
from src.visualizing.visualize import Visualize, PlotlyVisualize


try:
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{wasysym}')
except Exception as exc:
    pass


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
        self.metadata_version, self.grain_populations, self._grain_records = grain_analysis.read_grain_archive(
            'simdata/particle_params.h5')

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
            return self.sim.particles[rebound.hash(int(particle_hash))].params[param_name]
        except (AttributeError, KeyError, rebound.ParticleNotFound):
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

            if source_primary_hash is None:
                print("Assuming first object (star) as system primary.")
                return self.sim.particles[0]
            else:
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

            self.cached_timestep = None
            self.sim = self.sa[int(timestep)]
            _ = reboundx.Extras(self.sim, "simdata/rebx.bin")
        elif time is not None:
            self.cached_timestep = None
            self.sim = self.sa.getSimulation(time)
            _ = reboundx.Extras(self.sim, "simdata/rebx.bin")
        else:
            raise ValueError("timestep or time must be specified.")

        if self.reference_system is not None:
            self._rotate_reference_system()
            if timestep not in self.rotated_timesteps:
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
        self.grain_metadata = {
            field: np.zeros(self.sim.N, dtype='int64' if field in
                            ('particle_kind', 'grain_population_id', 'source_hash') else 'float64')
            for field in GRAIN_METADATA_FIELDS
        }
        self._snapshot_hashes = self.particle_hashes.copy()

        for k1 in range(self.sim.N):
            part_hash = int(self.particle_hashes[k1])
            grain_record = self._grain_records.get(part_hash)
            if grain_record is not None:
                for field in GRAIN_METADATA_FIELDS:
                    self.grain_metadata[field][k1] = grain_record[field]
                self.particle_weights[k1] = grain_record['grain_number_weight']
                self._particle_source_hashes[k1] = grain_record['source_hash']
                continue
            try:
                # Get parameters from h5 file with fallback to REBOUNDx
                part_creation_time = self.get_particle_param(part_hash, 'serpens_creation_time')
                species_id = self.get_particle_param(part_hash, 'serpens_species')
                source_hash = self.get_particle_param(part_hash, 'source_hash')

                if part_creation_time is None or species_id is None or source_hash is None:
                    continue

                species = [s for s in GLOBAL_PARAMETERS.get('all_species') if s.id == species_id][0]
                self.particle_species[k1] = species_id
                self.particle_weights[k1] = np.exp(
                    -(self.sim.t - part_creation_time) / species.tau #species.network
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
        self.cached_timestep = timestep

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
        overall_mask = (np.logical_or.reduce(masks_for_each_source) if masks_for_each_source
                        else np.zeros(len(self.particle_positions), dtype=bool))

        # Then apply the overall_mask to your arrays:
        self.particle_positions = self.particle_positions[overall_mask]
        self.particle_velocities = self.particle_velocities[overall_mask]
        self.particle_hashes = self.particle_hashes[overall_mask]
        self.particle_species = self.particle_species[overall_mask]
        self.particle_weights = self.particle_weights[overall_mask]
        self._particle_source_hashes = self._particle_source_hashes[overall_mask]
        self.grain_metadata = {field: values[overall_mask] for field, values in self.grain_metadata.items()}

    @ensure_data_loaded
    def grain_field(self, timestep, quantity='number', d=3, population_id=None, source_hash=None, radius_range=None):
        """Return physical DTFE vertex density and immutable weights in SI units.

        Quantities are number (W), mass (W m), and area (W pi a**2). The 2D
        field projects onto xy; the 3D field uses xyz. Selection follows analyzer
        cutoffs and inclusive grain-size radius_range in metres. Empty selections
        have estimator=None; insufficient/degenerate geometry raises ValueError.
        """
        mask = grain_analysis.grain_selection(self.grain_metadata, population_id, source_hash, radius_range)
        metadata = {field: values[mask] for field, values in self.grain_metadata.items()}
        return grain_analysis.estimate_field(self.particle_positions[mask], self.particle_velocities[mask],
                                             self.particle_hashes[mask], metadata, quantity, d)

    @ensure_data_loaded
    def grain_radial_profile(self, timestep, bins, quantity='number', center=None,
                             population_id=None, source_hash=None, radius_range=None):
        """Bin physical weights by cylindrical xy radius about center (default SI origin).

        bins is a positive count or explicit radius edges in metres. Returns
        integrated_weights per annulus and column_density per square metre,
        with units for both; no DTFE or survivor renormalization is involved.
        Analyzer cutoffs apply. A three-coordinate center ignores its z value.
        """
        mask = grain_analysis.grain_selection(self.grain_metadata, population_id, source_hash, radius_range)
        metadata = {field: values[mask] for field, values in self.grain_metadata.items()}
        return grain_analysis.radial_profile(self.particle_positions[mask], metadata, bins, quantity, center)

    @ensure_data_loaded
    def grain_budget(self, timestep, population_id=None, source_hash=None, radius_range=None):
        """Return number/mass/area birth, retention and loss totals, ignoring display cutoffs.

        Uses all historical records born before this snapshot, or present in it
        (disambiguating equal-time injections). Losses take effect at interval
        end, never for hashes still present in the snapshot. closure_residual is
        injected - retained - removed; unaccounted and its hashes explicitly
        report absent tracers without an effective recorded loss.
        """
        return grain_analysis.historical_budget(self._grain_records, self._snapshot_hashes, self.sim.t,
                                                population_id, source_hash, radius_range)

    def _los_visibility_mask(self, points):
        """Return visibility from positive x, using every positive-radius body's xyz.

        Points strictly behind a body's center and inside its projected yz disk
        are hidden. Points on the limb or center plane remain visible.
        """
        visible = np.ones(len(points), dtype=bool)
        for body in self.sim.particles:
            if getattr(body, 'r', 0) <= 0:
                continue
            projected_distance = np.sqrt((points[:, 1] - body.y) ** 2 +
                                         (points[:, 2] - body.z) ** 2)
            visible &= ~((projected_distance < body.r) & (points[:, 0] < body.x))
        return visible

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
            - dens: Array of density values at each point
            - delaunay: Triangulation, or None for an empty species selection

        Empty selections return an empty density array. Nonempty selections require
        positive injection and weight sums, and nondegenerate geometry. LOS requires d=2.
        """
        if isinstance(d, (bool, np.bool_)) or d not in (2, 3):
            raise ValueError("DTFE dimension must be 2 or 3.")
        if not isinstance(los, (bool, np.bool_)) or (los and d != 2):
            raise ValueError("DTFE los must be a boolean; LOS estimation requires dimension d=2.")
        d = int(d)
        points_mask = np.where(self.particle_species == species.id)

        points = self.particle_positions[points_mask]
        velocities = self.particle_velocities[points_mask]
        weights = self.particle_weights[points_mask]
        if len(points) == 0:
            return np.empty(0, dtype=float), None

        # Physical weight calculation:
        total_injected = timestep * (species.n_sp + species.n_th)
        if not np.isfinite(total_injected) or total_injected <= 0:
            raise ValueError("Gas DTFE requires positive total injected particles; use a positive timestep "
                             "and species.n_sp + species.n_th > 0.")
        weight_sum = np.sum(weights)
        if not np.all(np.isfinite(weights)) or not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Gas DTFE requires finite weights with a positive weight sum; "
                             "check particle lifetimes and the selected timestep.")
        remaining_part = len(points[:, 0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * self.sim.t
        number_of_particles = mass_in_system / species.m
        phys_weights = number_of_particles * weights/weight_sum

        coordinates = points[:, 1:3] if los else points[:, :d]
        projected_velocities = velocities[:, 1:3] if los else velocities[:, :d]
        if len(points) < d + 1:
            raise ValueError(f"Gas DTFE in {d}D requires at least {d + 1} points; got {len(points)}. "
                             "Select a later timestep or relax particle cutoffs.")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(projected_velocities)):
            raise ValueError("Gas DTFE requires finite positions and velocities; check the selected snapshot.")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError(f"Gas DTFE has duplicate positions in {d}D; select a different projection "
                             "or timestep. No jitter is applied.")
        if np.linalg.matrix_rank(coordinates - coordinates[0]) < d:
            raise ValueError(f"Gas DTFE geometry is degenerate in {d}D; select a different dimension, "
                             "projection or timestep. No jitter is applied.")
        try:
            dtfe = (DTFE if d == 2 else DTFE3D).DTFE(coordinates, projected_velocities, phys_weights)
            dens = dtfe.density(*coordinates.T) / (1e4 if d == 2 else 1e6)
        except (QhullError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
            raise ValueError(f"Gas DTFE geometry is degenerate or numerically ill-conditioned in {d}D; "
                             "select a different projection or timestep. No jitter is applied.") from exc
        if not np.all(np.isfinite(dens)):
            raise ValueError(f"Gas DTFE produced nonfinite density in {d}D; check geometry and physical weights.")
        if los:
            dens[~self._los_visibility_mask(points)] = 0

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
                timestep, species, d=(2 if is_los else d), los=is_los,
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
            vis.set_title(fr"Particle Densities $log_{{10}} (N/\mathrm{{cm}}^{{{-d}}})$", size=25, color='white')

    def _add_los_scatter(self, vis, points, dens, species_index, **kwargs):
        """Add line of sight scatter plot with planet masking."""
        # Auto-adjust visualization levels if specified
        if kwargs.get('lvlmin') == 'auto':
            vis.vis_params.update({"lvlmin": np.log10(np.min(dens[dens > 0])) - 0.5})
        if kwargs.get('lvlmax') == 'auto':
            vis.vis_params.update({"lvlmax": np.log10(np.max(dens[dens > 0])) + 0.5})

        # Apply the same body masking used by density estimation
        mask = self._los_visibility_mask(points)

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

    @staticmethod
    def _profile_species(species_num):
        species = GLOBAL_PARAMETERS.get('all_species', [])
        if isinstance(species_num, (bool, np.bool_)) or not isinstance(species_num, (int, np.integer)) \
                or not 1 <= species_num <= len(species):
            raise ValueError('species_num must be a one-based index of an available gas species.')
        return species[species_num - 1]

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
        from src.visualizing.data import PlotRequest, DisplayOptions, VisualizationDataProvider
        from src.visualizing.figures import build_figure

        species = self._profile_species(species_num)
        if getattr(self, '_plot_data_provider', None) is None:
            self._plot_data_provider = VisualizationDataProvider(self)
        request = PlotRequest(timestep=timestep, view='3d', dimension=3, species_ids=(species.id,))
        display = DisplayOptions(thresholds={str(species.id): (log_cutoff, None)}, show_star=show_star)
        return build_figure(self._plot_data_provider.prepare(request), display)

    @ensure_data_loaded
    def calculate_profile(self, timestep, species_num=1, max_distance_rp=6, source_hash=None):
        """Return the original 100-sample yz cut and sigma=3 Gaussian smoothing.

        Radii are in source radii and densities in cm^-2. The source selects the
        cut origin only; all particles of the selected species contribute with
        the legacy gas normalization, without vertex occultation masking.
        """
        species = self._profile_species(species_num)
        if isinstance(max_distance_rp, (bool, np.bool_)):
            raise ValueError('max_distance_rp must be finite and at least 1.')
        try:
            max_distance_rp = float(max_distance_rp)
        except (TypeError, ValueError) as exc:
            raise ValueError('max_distance_rp must be finite and at least 1.') from exc
        if not np.isfinite(max_distance_rp) or max_distance_rp < 1:
            raise ValueError('max_distance_rp must be finite and at least 1.')
        if not self.source_hashes:
            raise ValueError('Profile requires an available particle source.')
        if source_hash is None:
            source_hash = self.source_hashes[0]
        try:
            if isinstance(source_hash, (int, np.integer)) or \
                    isinstance(source_hash, str) and source_hash.isdecimal():
                source_hash = rebound.hash(int(source_hash))
            source = self.sim.particles[source_hash]
            if source.hash.value not in {self.sim.particles[h].hash.value for h in self.source_hashes}:
                raise ValueError('Selected body is not a particle source.')
        except (rebound.ParticleNotFound, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError('Select an available particle source by name or hash.') from exc
        if not np.isfinite(source.r) or source.r <= 0:
            raise ValueError('Profile requires a finite positive source radius.')
        if not np.all(np.isfinite([source.y, source.z])):
            raise ValueError('Profile requires finite source coordinates.')

        points_mask = np.where(self.particle_species == species.id)
        points = self.particle_positions[points_mask]
        velocities = self.particle_velocities[points_mask]
        weights = self.particle_weights[points_mask]
        if len(points) < 3:
            raise ValueError(f'Gas profile DTFE requires at least 3 points; got {len(points)}. '
                             'Select a later timestep or relax particle cutoffs.')
        if not np.all(np.isfinite(points[:, 1:3])) or not np.all(np.isfinite(velocities[:, 1:3])):
            raise ValueError('Gas profile DTFE requires finite positions and velocities.')
        if len(np.unique(points[:, 1:3], axis=0)) != len(points):
            raise ValueError('Gas profile DTFE has duplicate yz positions; select another timestep. '
                             'No jitter is applied.')
        if np.linalg.matrix_rank(points[:, 1:3] - points[0, 1:3]) < 2:
            raise ValueError('Gas profile DTFE geometry is degenerate in yz; select another timestep. '
                             'No jitter is applied.')

        # Physical weight calculation:
        total_injected = timestep * (species.n_sp + species.n_th)
        if not np.isfinite(total_injected) or total_injected <= 0:
            raise ValueError('Gas profile requires positive total injected particles; use a positive timestep '
                             'and species.n_sp + species.n_th > 0.')
        weight_sum = np.sum(weights)
        if not np.all(np.isfinite(weights)) or not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError('Gas profile requires finite weights with a positive weight sum; '
                             'check particle lifetimes and the selected timestep.')
        if not np.isfinite(species.m) or species.m <= 0:
            raise ValueError('Gas profile requires a finite positive species particle mass.')
        remaining_part = len(points[:, 0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * self.sim.t
        number_of_particles = mass_in_system / species.m
        phys_weights = number_of_particles * weights / weight_sum
        if not np.all(np.isfinite(phys_weights)):
            raise ValueError('Gas profile requires finite physical weights; check species and snapshot parameters.')

        # 1D CUT
        x_linspace = np.linspace(source.y + source.r, source.y + max_distance_rp*source.r, 100)
        y_fixed = np.full(100, source.z)
        if not np.all(np.isfinite(x_linspace)):
            raise ValueError('Profile sampling coordinates must be finite; reduce max_distance_rp.')
        try:
            dtfe = DTFE.DTFE(points[:, 1:3], velocities[:, 1:3], phys_weights)
            dens_cut = dtfe.density(x_linspace, y_fixed) / 1e4
        except (QhullError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
            raise ValueError('Gas profile DTFE geometry is degenerate or numerically ill-conditioned; '
                             'select another timestep. No jitter is applied.') from exc
        if not np.all(np.isfinite(dtfe.rho)) or not np.all(np.isfinite(dens_cut)):
            raise ValueError('Gas profile DTFE produced nonfinite density; check geometry and physical weights.')
        dens_cut_smooth = gaussian_filter1d(dens_cut, sigma=3)
        return dict(radii=(x_linspace - source.y) / source.r, raw=dens_cut,
                    smoothed=dens_cut_smooth, units='cm^-2', name=species.name)

    def plot_1d_cut(self, timestep, species_num=1, log_scale=False, max_distance_rp=6, **kwargs):
        profile = self.calculate_profile(timestep, species_num=species_num, max_distance_rp=max_distance_rp)
        plt.figure(dpi=200, **kwargs)
        plt.plot(profile['radii'], profile['raw'])
        plt.plot(profile['radii'], profile['smoothed'], c='r')
        if log_scale:
            plt.yscale("log")
        plt.xlabel(r"$R_P$")
        plt.ylabel(r"$N\ [\mathrm{cm}^{-2}]$")
        plt.tight_layout()
        plt.show()

    def _phase_source_orbit(self, source_name):
        try:
            if isinstance(source_name, (bool, np.bool_)):
                raise ValueError('Boolean source selection is not supported.')
            if isinstance(source_name, (int, np.integer)) or \
                    isinstance(source_name, str) and source_name.isdecimal():
                source_name = rebound.hash(int(source_name))
            if not isinstance(source_name, (str, c_uint)):
                raise ValueError('Expected a source name or hash.')
            source_particle = self.sim.particles[source_name]
        except (rebound.ParticleNotFound, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError('Select an available phase source by name or integer hash, not particle index.') from exc
        try:
            source_primary_hash = self.get_particle_param(source_particle.hash.value, 'source_primary')
            source_primary = self.sim.particles[rebound.hash(int(source_primary_hash))]
            source_orbit = source_particle.orbit(primary=source_primary)
        except (rebound.ParticleNotFound, KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Phase source requires an available source_primary and a valid orbit; '
                             'check source metadata.') from exc
        if not np.isfinite(source_orbit.P) or source_orbit.P <= 0 or not np.isfinite(source_orbit.theta):
            raise ValueError('Phase source requires a finite positive orbital period and finite phase.')
        return source_orbit

    def calculate_phase_data(self, source_name, species_name=None, orbits=1):
        """Return gas phase statistics without saving or displaying them.

        Actual archive indices in the final orbital time window are used, excluding
        timestep zero (no injected particles). Windows extending before the archive
        use the available snapshots. Statistics retain the legacy clipped log10
        density definition, including occulted particles as zero-valued samples.
        """
        original_reference_system = self.reference_system
        self.reference_system = None

        all_species = GLOBAL_PARAMETERS.get('all_species', [])
        if not all_species:
            raise ValueError('Phase calculation requires an available gas species.')
        if species_name is None:
            species = all_species[0]
        else:
            if not isinstance(species_name, str):
                raise ValueError('Select a gas species by name.')
            matches = [s for s in all_species if s.name == species_name]
            if len(matches) != 1:
                raise ValueError(f'Select an available, unambiguous gas species; got {species_name!r}.')
            species = matches[0]
        injection = species.n_sp + species.n_th
        if not np.isfinite(injection) or injection <= 0:
            raise ValueError('Phase calculation requires positive particle injection; '
                             'set species.n_sp + species.n_th > 0.')
        try:
            if isinstance(orbits, (bool, np.bool_)):
                raise ValueError
            orbits = float(orbits)
            if not np.isfinite(orbits) or orbits <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError('orbits must be finite and positive.') from exc

        if getattr(self, 'sa', None) is None or len(self.sa) < 2:
            raise ValueError('Phase calculation requires an archive with at least two snapshots; '
                             'run the simulation beyond timestep zero.')
        try:
            times = np.array([self.sa[index].t for index in range(len(self.sa))], dtype=float)
        except (OSError, IndexError, ValueError) as exc:
            raise ValueError('Cannot read phase archive snapshots; check the simulation archive.') from exc
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0) or times[-1] <= times[0]:
            raise ValueError('Phase archive times must be finite, nondecreasing, and span a positive duration.')
        self.load_timestep_data(timestep=len(self.sa) - 1)
        source_orbit = self._phase_source_orbit(source_name)
        orbit_start_time = times[-1] - orbits * source_orbit.P
        timesteps = [index for index, time in enumerate(times) if index > 0 and time >= orbit_start_time]

        phase_data = []
        for t in timesteps:
            self.load_timestep_data(timestep=t)
            phase = self._phase_source_orbit(source_name).theta * 180 / np.pi
            statistics = []
            for dimension in (2, 3):
                try:
                    if dimension == 2:
                        density, _ = self.delaunay_field_estimation(t, species, d=2, los=True)
                    else:
                        density, _ = self.delaunay_field_estimation(t, species, d=3)
                except ValueError as exc:
                    raise ValueError(f'Phase timestep {t}, {dimension}D: {exc}') from exc
                density = np.asarray(density)
                if density.size == 0:
                    raise ValueError(f'Phase timestep {t}, {dimension}D has no gas particles; '
                                     'select a later window/species or relax particle cutoffs.')
                if not np.all(np.isfinite(density)) or np.any(density < 0):
                    raise ValueError(f'Phase timestep {t}, {dimension}D requires finite nonnegative density; '
                                     'check sample geometry and physical weights.')
                log_density = np.log10(density + 1e-5)
                log_density[log_density < 0] = 0
                statistics.extend((np.max(log_density), np.mean(log_density)))
            phase_data.append([phase, t, *statistics])

        # restore reference system
        self.reference_system = original_reference_system
        return pd.DataFrame(
            data=phase_data,
            columns=['Phase', 'Timestep', 'Max_2D', 'Mean_2D', 'Max_3D', 'Mean_3D']
        )

    def calculate_phasecurve(self, source_name, species_name=None, orbits=1):
        df = self.calculate_phase_data(source_name, species_name=species_name, orbits=orbits)
        df.to_csv('phase-curve.csv', index=False)

    @staticmethod
    def plot_phasecurve(filename='phase-curve.csv', column_density=True, particle_density=True, type="max"):
        from src.visualizing.figures import build_phase_figure

        df_phase = pd.read_csv(filename)
        fig = build_phase_figure(df_phase, statistic=type, column_density=column_density,
                                 particle_density=particle_density)
        return fig
