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
from datetime import datetime
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch

from src import DTFE, DTFE3D
from src.parameters import Parameters
from src.visualize import Visualize

matplotlib.use('TkAgg')
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amssymb}')


def ensure_data_loaded(method):
    """
    Used as decorator inside the 'SerpensAnalyzer' class.
    Makes sure that the correct data is loaded for a given timestep.
    """
    def wrapper(self, timestep, *args, **kwargs):
        self.pull_data(timestep)
        return method(self, timestep, *args, **kwargs)
    return wrapper


class SerpensAnalyzer:
    """
    This is the SERPENS analyzer class.
    It utilizes Delaunay Field Density Estimations to calculate particle densities from previously run
    SERPENS simulations.
    It also provides the main interface for plotting.
    """
    def __init__(self, save_output=False, save_archive=False, folder_name=None,
                 z_cutoff=None, r_cutoff=None, v_cutoff=None, reference_system="heliocentric"):
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
        reference_system : str      (default: "heliocentric")
            Can either be "heliocentric" or "geocentric". Latter will rotate the system, such that the (exo-)planet
            keeps y-coordinate 0. This is equal to a co-rotating observer.
        """

        try:
            with open('Parameters.pkl', 'rb') as handle:
                params_load = pickle.load(handle)
                params_load()
        except Exception:
            raise Exception("Parameters.pickle not found.")

        self.sa = self._load_simulation_archive()
        self.save = save_output
        self.save_arch = save_archive
        self.save_index = 1

        self.params = Parameters()

        self._sim_instance = None
        self._particle_positions = None
        self._particle_velocities = None
        self._particle_hashes = None
        self._particle_species = None
        self._particle_weights = None
        self.cached_timestep = None
        self.rotated_timesteps = []

        self.cutoffs = {"z": z_cutoff, "r": r_cutoff, "v": v_cutoff}
        self.reference_system = reference_system

        if save_output:
            print("Copying and saving...")
            if folder_name is None:
                self.path = datetime.utcnow().strftime("%d%m%Y--%H-%M-%S")
            else:
                self.path = folder_name

            if not os.path.exists(f'output/{self.path}/plots'):
                os.makedirs(f'output/{self.path}/plots')

            try:
                shutil.copy2(f'{os.getcwd()}/Parameters.txt', f'output/{self.path}/Parameters.txt')
            except:
                pass

            if save_archive:
                print("\t archive...")
                shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/output/{self.path}")

    @staticmethod
    def _load_simulation_archive():
        """
        Loads simulation archive.bin file.
        See https://rebound.readthedocs.io/en/latest/ipython_examples/SimulationArchive/ for more information.
        """
        try:
            return rebound.Simulationarchive("archive.bin")
        except Exception:
            raise Exception("simulation archive not found.")

    def _calculate_offsets(self, plane):
        """
        Internal use only.
        Returns coordinate offsets if the primary is a planet (source is moon).
        Arguments handled by class functions.
        """
        #if self.source_is_moon:
        #    planet = self._sim_instance.particles["planet"]
        #    if plane == 'xy':
        #        return planet.x, planet.y, 0
        #    elif plane == 'yz':
        #        return planet.y, planet.z, 0
        #    elif plane == '3d':
        #        return planet.x, planet.y, planet.z
        #else:
        #    return 0, 0, 0
        planet = self.get_primary(0)
        if plane == 'xy':
            return planet.x, planet.y, 0
        elif plane == 'yz':
            return planet.y, planet.z, 0
        elif plane == '3d':
            return planet.x, planet.y, planet.z

    def _rotate_reference_system(self):
        """
        Internal use only.
        Applies a rotation (coordinate transformation) to all particles if the reference system is geocentric.
        """
        #if self.source_is_moon:
        phase = np.arctan2(self.get_primary(0).y, self.get_primary(0).x)

        inc = self.get_primary(0).orbit().inc

        reb_rot = rebound.Rotation(angle=phase, axis='z')
        reb_rot_inc = rebound.Rotation(angle=inc, axis='y')
        for particle in self._sim_instance.particles:
            particle.rotate(reb_rot.inverse())
            particle.rotate(reb_rot_inc)
        #else:
        #    pass

    def get_primary(self, source_index) -> rebound.Particle:
        return self._sim_instance.particles[
            rebound.hash(self._sim_instance.particles[f"source{source_index}"].params['source_primary'])]

    def pull_data(self, timestep):
        """
        Serializes particle vectors and attributes for a given timestep.
        Sets REBOUND simulation instance at given timestep.
        Gets called by the @ensure_data_loaded decorator.
        """
        if self.cached_timestep == timestep:
            return
        else:
            self.cached_timestep = timestep

        self._sim_instance = self.sa[int(timestep)]
        _ = reboundx.Extras(self._sim_instance, "rebx.bin")

        if self.reference_system == "geocentric":
            if timestep not in self.rotated_timesteps:
                self._rotate_reference_system()
                self.rotated_timesteps.append(timestep)

        self._particle_positions = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._particle_velocities = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._particle_hashes = np.zeros(self._sim_instance.N, dtype="uint32")
        self._sim_instance.serialize_particle_data(xyz=self._particle_positions, vxvyvz=self._particle_velocities,
                                                   hash=self._particle_hashes)

        self._particle_species = np.zeros(self._sim_instance.N, dtype="int")
        self._particle_weights = np.zeros(self._sim_instance.N, dtype="float64")
        for k1 in range(self._sim_instance.N):
            try:
                self._particle_species[k1] = self._sim_instance.particles[
                    rebound.hash(int(self._particle_hashes[k1]))].params["serpens_species"]
                self._particle_weights[k1] = self._sim_instance.particles[
                    rebound.hash(int(self._particle_hashes[k1]))].params["serpens_weight"]
            except AttributeError:
                continue

        self._apply_masks()

    def _apply_masks(self):
        """
        Internal use only.
        Apply filters to particles. Removes all particles according to z_cutoff, r_cutoff, and v_cutoff.
        """
        combined_mask = np.ones(len(self._particle_positions), dtype=bool)
        for cutoff_type in self.cutoffs:
            cutoff_value = self.cutoffs[cutoff_type]

            if cutoff_value is not None:
                assert isinstance(cutoff_value, (float, int))
                condition = None

                if cutoff_type == "z":
                    condition = (self._particle_positions[:, 2] < cutoff_value * self.get_primary(0).r) & \
                                (self._particle_positions[:, 2] > -cutoff_value * self.get_primary(0).r)
                elif cutoff_type == "r":
                    r = np.linalg.norm(self._particle_positions - self.get_primary(0).xyz, axis=1)
                    condition = r < cutoff_value * self.get_primary(0).r
                elif cutoff_type == "v":
                    v = np.linalg.norm(self._particle_velocities - self._sim_instance.particles["source0"].vxyz, axis=1)
                    condition = v < cutoff_value

                if condition is not None:
                    combined_mask &= condition

        self._particle_positions = self._particle_positions[combined_mask]
        self._particle_velocities = self._particle_velocities[combined_mask]
        self._particle_hashes = self._particle_hashes[combined_mask]
        self._particle_species = self._particle_species[combined_mask]
        self._particle_weights = self._particle_weights[combined_mask]

    @ensure_data_loaded
    def delaunay_field_estimation(self, timestep, species, d=2, los=False):
        """
        Main function to get density values by initializing the DTFE estimator at a certain timestep.
        Automatically runs the "pull_data" function through the decorator.
        Restricts itself to a single species given as an argument.
        Can be used as a 3D particle density estimator or 2D line of sight density estimator.
        By default, this function initializes a grid on which densities are calculated.

        Arguments
        ---------
        timestep : int
            Simulation timestep at which to calculate densities.
        species : Species class instance
            Particle species to be analyzed.
        d : int     (default: 2)
            Dimension of analysis.
            For d=2 the returning densities will be particles per cm^2.
            For d=3 the returning densities will be particles per cm^3.
        los : bool      (default: False)
            If 'True', we assume that we look at the system from the positive x-axis (line of sight).
            A large quantity of particles may be hidden behind the source's primary from this POV.
            Therefore, apply mask to filter out particles behind the primary before calculating densities if 'True'.
            Important to set to 'False' if we consider looking on the orbital plane. Particles won't be masked as they
            are not hidden.
        """
        simulation_time = (timestep * self.params.int_spec["sim_advance"] *
                           self._sim_instance.particles["source0"].orbit(
                              primary=self.get_primary(0)).P)

        points_mask = np.where(self._particle_species == species.id)

        points = self._particle_positions[points_mask]
        velocities = self._particle_velocities[points_mask]
        weights = self._particle_weights[points_mask]

        # Filter points:
        indices = np.unique([tuple(row) for row in points], axis=0, return_index=True)[1]
        weights = weights[np.sort(indices)]
        points = points[np.sort(indices)]
        velocities = velocities[np.sort(indices)]

        # Physical weight calculation:
        total_injected = timestep * (species.n_sp + species.n_th)
        remaining_part = len(points[:, 0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
        number_of_particles = mass_in_system / species.m
        phys_weights = number_of_particles * weights/np.sum(weights)

        if d == 2:

            if los:
                los_dist_to_planet = np.sqrt((points[:, 1] - self.get_primary(0).y) ** 2 +
                                             (points[:, 2] - self.get_primary(0).z) ** 2)
                mask = ((los_dist_to_planet < self.get_primary(0).r) &
                        (points[:, 0] - self.get_primary(0).x < 0))

                dtfe = DTFE.DTFE(points[:, 1:3], velocities[:, 1:3], phys_weights)

                dens = dtfe.density(points[:, 1], points[:, 2]) / 1e4
                dens[mask] = 0

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
        return self._particle_positions, self._particle_velocities

    def plot_planar(self, timestep, d=3, scatter=True, triplot=False, show=True, **kwargs):
        """
        Returns a plot of the system looking at the orbital plane (top-down view).
        Accesses the 'Visualizer' class to construct the plots and forwards keyword arguments.
        
        Arguments
        ---------
        timestep : int
            Timestep at which to create the plot.
        d : int     (default: 3)
            Dimension of density calculation.
            For d=2 the returning density plot will depict particles per cm^2.
            For d=3 the returning density plot will be particles per cm^3.
        scatter : bool      (default: True)
            Whether to create a scatter plot. Scatters are defined by particle positions and their color represents
            density as calculated with the DTFE in dimension d.
        triplot : bool      (default: False)
            Whether to plot the delaunay tessellation.
        show : bool     (default: True)
            Whether to show the plot. If 'False' make sure to set the 'save_output' argument of the analyzer
            initialization is 'True'.
        kwargs : Keyword arguments
            Passed to Visualizer (see src/visualize.py)
        """

        ts_list = np.atleast_1d(timestep).astype(int)

        for ts in ts_list:
            self.pull_data(ts)
            vis = Visualize(self._sim_instance, **kwargs)

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)
                points = self._particle_positions[np.where(self._particle_species == species.id)]

                if len(points) == 0:
                    vis.empty(k)
                    continue

                dens, delaunay = self.delaunay_field_estimation(ts, species, d=d)

                if scatter:
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, d=d)

                if triplot:
                    if d == 3:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices[:,:3])
                    elif d == 2:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices)

                vis.set_title(fr"Particle Densities $log_{{10}} (N/\mathrm{{cm}}^{{{-d}}})$ around Planetary Body", size=25, color='w')

            if self.save:
                vis(show_bool=show, save_path=self.path, filename=f'TD_{ts}_000{self.save_index}')
                self.save_index += 1

                # Handle saving bugs...
                list_of_files = glob.glob(f'./output/{self.path}/plots/*')
                latest_file = max(list_of_files, key=os.path.getctime)
                if os.path.getsize(latest_file) < 50000:
                    print("\t Detected low filesize (threshold at 50 KB). Possibly encountered a saving bug. Retrying process.")
                    os.remove(latest_file)

            else:
                vis(show_bool=show)

            del vis

    def plot_lineofsight(self, timestep, show=False, scatter=True, **kwargs):
        """
        Returns a plot of the system from a line of sight perspective.
        Accesses the 'Visualizer' class to construct the plots and forwards keyword arguments.

        Arguments
        ---------
        timestep : int
            Timestep at which to create the plot.
        colormesh : bool    (default: False)
            Whether to plot a colormesh (currently not available).
        scatter : bool      (default: True)
            Whether to create a scatter plot. Scatters are defined by particle positions and their color represents
            density as calculated with the DTFE in dimension d.
        show : bool     (default: True)
            Whether to show the plot. If 'False' make sure to set the 'save_output' argument of the analyzer
            initialization is 'True'.
        kwargs : Keyword arguments
            Passed to Visualizer (see src/visualize.py)
        """

        ts_list = np.atleast_1d(timestep).astype(int)

        running_index = 0
        while running_index < len(ts_list):
            ts = ts_list[::-1][running_index]
            self.pull_data(ts)

            vis = Visualize(self._sim_instance, perspective='los', **kwargs)

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)

                if scatter:
                    points = self._particle_positions[np.where(self._particle_species == species.id)]
                    dens, delaunay = self.delaunay_field_estimation(ts, species, d=2, los=True)

                    if running_index == 0:
                        if kwargs.get('lvlmin', None) == 'auto':
                            vis.vis_params.update({"lvlmin": np.log10(np.min(dens[dens > 0])) - .5})
                        if kwargs.get('lvlmax', None) == 'auto':
                            vis.vis_params.update({"lvlmax": np.log10(np.max(dens[dens > 0])) + .5})

                    los_dist_to_planet = np.sqrt((points[:, 1] - self.get_primary(0).y) ** 2 +
                                                 (points[:, 2] - self.get_primary(0).z) ** 2)
                    mask = (los_dist_to_planet > self.get_primary(0).r) | \
                           (points[:, 0] - np.abs(self.get_primary(0).x) > 0)
                    vis.add_densityscatter(k, -points[:, 1][mask], points[:, 2][mask], dens[mask], d=2)

            if self.save:
                vis(show_bool=show, save_path=self.path, filename=f'LOS_{ts}_{self.save_index}')
                self.save_index += 1

                # Handle saving bugs...
                list_of_files = glob.glob(f'./output/{self.path}/plots/*')
                latest_file = max(list_of_files, key=os.path.getctime)
                if os.path.getsize(latest_file) < 50000:
                    print(
                        "\t Detected low filesize (threshold at 50 KB). Possibly encountered a saving bug. Retrying process.")
                    os.remove(latest_file)
                else:
                    running_index += 1
            else:
                vis(show_bool=show)
                running_index += 1

            del vis

    @ensure_data_loaded
    def plot3d(self, timestep, species_num=1, log_cutoff=None, show_star=False):
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
        species = self.params.get_species(num=species_num)
        pos = self._particle_positions[np.where(self._particle_species == species.id)]
        dens, _ = self.delaunay_field_estimation(timestep, species, d=3)

        phi, theta = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
        x_p = (self.get_primary(0).r * np.sin(theta) * np.cos(phi) +
               self.get_primary(0).x)
        y_p = (self.get_primary(0).r * np.sin(theta) * np.sin(phi) +
               self.get_primary(0).y)
        z_p = (self.get_primary(0).r * np.cos(theta) +
               self.get_primary(0).z)

        if show_star:
            x_s = (self._sim_instance.particles["star"].r * np.sin(theta) * np.cos(phi) +
                   self._sim_instance.particles["star"].x)
            y_s = (self._sim_instance.particles["star"].r * np.sin(theta) * np.sin(phi) +
                   self._sim_instance.particles["star"].y)
            z_s = (self._sim_instance.particles["star"].r * np.cos(theta) +
                   self._sim_instance.particles["star"].z)

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

        fig.add_trace(go.Surface(x=x_p, y=y_p, z=z_p, surfacecolor=np.zeros(shape=x_p.shape), showscale=False, colorscale='matter'))
        if show_star:
            fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, surfacecolor=np.zeros(shape=x_p.shape), showscale=False, colorscale='Hot'))
        fig.update_coloraxes(colorbar_exponentformat='e')
        fig.update_layout(scene_aspectmode='data')
        fig.show()

        np.seterr(divide='warn')
