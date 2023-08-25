import numpy as np
import os as os
import glob
import shutil
import rebound
import dill
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src import DTFE, DTFE3D
from src.parameters import Parameters
from src.visualize import Visualize

matplotlib.use('TkAgg')


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

    def __init__(self, save_output=False, save_archive=False, folder_name=None, z_cutoff=None, r_cutoff=None, reference_system="heliocentric", v_cutoff=None):

        try:
            with open('Parameters.pickle', 'rb') as handle:
                params_load = dill.load(handle)
                params_load()
        except Exception:
            raise Exception("hash_library.pickle and/or Parameters.pickle not found.")

        self.hash_supdict = self._load_hash_supdict()
        self.sa = self._load_simulation_archive()
        self.save = save_output
        self.save_arch = save_archive
        self.save_index = 1

        self.params = Parameters()
        self.moon_exists = self.params.int_spec["moon"]

        self._sim_instance = None
        self._p_positions = None
        self._p_velocities = None
        self._p_hashes = None
        self._p_species = None
        self._p_weights = None
        self.cached_timestep = None
        self.rotated_timesteps = []

        self.cutoffs = {"z": z_cutoff, "r": r_cutoff, "v": v_cutoff}
        self.reference_system = reference_system

        if save_output:
            print("Copying and saving...")
            if folder_name is None:
                if self.moon_exists:
                    self.path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_moonsource"
                else:
                    self.path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_planetsource"
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
                print("\t hash library...")
                shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/output/{self.path}")
                print("\t ...done!")

    @staticmethod
    def _load_hash_supdict():
        hash_supdict = {}
        with open('hash_library.pickle', 'rb') as f:
            while True:
                try:
                    a = dill.load(f)
                    dict_timestep = list(a.keys())[0]
                except EOFError:
                    break
                else:
                    hash_supdict[dict_timestep] = a[dict_timestep]
        return hash_supdict

    @staticmethod
    def _load_simulation_archive():
        try:
            return rebound.SimulationArchive("archive.bin", process_warnings=False)
        except Exception:
            raise Exception("simulation archive not found.")

    @ensure_data_loaded
    def __grid(self, timestep, plane='xy'):
        boundary = self.params.int_spec["r_max"] * self._get_orbit_a_factor()
        offsetx, offsety, offsetz = self._calculate_offsets(plane)

        if plane == '3d':
            X, Y, Z = np.meshgrid(np.linspace(-boundary + offsetx, boundary + offsetx, 100),
                                  np.linspace(-boundary + offsety, boundary + offsety, 100),
                                  np.linspace(-boundary + offsetz, boundary + offsetz, 100))
            return X, Y, Z
        else:
            X, Y = np.meshgrid(np.linspace(-boundary + offsetx, boundary + offsetx, 300),
                               np.linspace(-boundary + offsety, boundary + offsety, 300))
            return X, Y

    def _get_orbit_a_factor(self):
        if self.moon_exists:
            return self._sim_instance.particles["moon"].calculate_orbit(
                primary=self._sim_instance.particles["planet"]).a
        else:
            return self._sim_instance.particles["planet"].a

    def _calculate_offsets(self, plane):
        if self.moon_exists:
            planet_x, planet_y, planet_z = self._sim_instance.particles["planet"].x, self._sim_instance.particles[
                "planet"].y, self._sim_instance.particles["planet"].z
            if plane == 'xy':
                return planet_x, planet_y, 0
            elif plane == 'yz':
                return planet_y, planet_z, 0
            elif plane == '3d':
                return planet_x, planet_y, planet_z
        else:
            return 0, 0, 0

    def _rotate_reference_system(self):
        phase = np.arctan2(self._sim_instance.particles["planet"].y, self._sim_instance.particles["planet"].x)

        inc = self._sim_instance.particles["planet"].calculate_orbit().inc

        reb_rot = rebound.Rotation(angle=phase, axis='z')
        reb_rot_inc = rebound.Rotation(angle=inc, axis='y')
        for particle in self._sim_instance.particles:
            particle.rotate(reb_rot.inverse())
            particle.rotate(reb_rot_inc)

    def pull_data(self, timestep):

        if self.cached_timestep == timestep:
            return
        else:
            self.cached_timestep = timestep

        # REBX: sim_instance, rebx = self.sa[timestep]
        self._sim_instance = self.sa[timestep]

        if self.reference_system == "geocentric":
            if timestep not in self.rotated_timesteps:
                self._rotate_reference_system()
                self.rotated_timesteps.append(timestep)

        self._p_positions = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._p_velocities = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._p_hashes = np.zeros(self._sim_instance.N, dtype="uint32")
        self._sim_instance.serialize_particle_data(xyz=self._p_positions, vxvyvz=self._p_velocities,
                                                   hash=self._p_hashes)

        if not timestep == 0:
            hash_dict_current = self.hash_supdict[str(timestep)]
        else:
            hash_dict_current = {}

        self._p_species = np.zeros(self._sim_instance.N, dtype="int")
        self._p_weights = np.zeros(self._sim_instance.N, dtype="float64")
        for k1 in range(self._sim_instance.N_active, self._sim_instance.N):
            self._p_species[k1] = hash_dict_current[str(self._p_hashes[k1])]["id"]
            self._p_weights[k1] = hash_dict_current[str(self._p_hashes[k1])]["weight"]

            # REBX:
            # try:
            #     self._p_species[k1] = sim_instance.particles[rebound.hash(int(self._p_hashes[k1]))].params["serpens_species"]
            #     self._p_weights[k1] = sim_instance.particles[rebound.hash(int(self._p_hashes[k1]))].params["serpens_weight"]
            # except AttributeError:
            #     self._p_species[k1] = 0
            #     self._p_weights[k1] = 0
            #     print("Particle not weightable")

        self._apply_masks()

    def _apply_masks(self):

        combined_mask = np.ones(len(self._p_positions), dtype=bool)
        for cutoff_type in self.cutoffs:
            cutoff_value = self.cutoffs[cutoff_type]

            if cutoff_value is not None:
                assert isinstance(cutoff_value, (float, int))
                condition = None

                if cutoff_type == "z":
                    condition = (self._p_positions[:, 2] < cutoff_value * self._sim_instance.particles["planet"].r) & \
                                (self._p_positions[:, 2] > -cutoff_value * self._sim_instance.particles["planet"].r)
                elif cutoff_type == "r":
                    r = np.linalg.norm(self._p_positions - self._sim_instance.particles["planet"].xyz, axis=1)
                    condition = r < cutoff_value * self._sim_instance.particles["planet"].r
                elif cutoff_type == "v":
                    v = np.linalg.norm(self._p_velocities - self._sim_instance.particles["moon"].vxyz, axis=1)
                    condition = v < cutoff_value

                if condition is not None:
                    combined_mask &= condition

        self._p_positions = self._p_positions[combined_mask]
        self._p_velocities = self._p_velocities[combined_mask]
        self._p_hashes = self._p_hashes[combined_mask]
        self._p_species = self._p_species[combined_mask]
        self._p_weights = self._p_weights[combined_mask]

    @ensure_data_loaded
    def dtfe(self, timestep, species, d=2, grid=True, los=False):

        if self.moon_exists:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * self._sim_instance.particles["moon"].calculate_orbit(primary=self._sim_instance.particles["planet"]).P
        else:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * self._sim_instance.particles["planet"].P

        points_mask = np.where(self._p_species == species.id)

        points = self._p_positions[points_mask]
        velocities = self._p_velocities[points_mask]
        weights = self._p_weights[points_mask]

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

        dtfe_obj = None

        if d == 2:
            if los:
                los_dist_to_planet = np.sqrt((points[:, 1] - self._sim_instance.particles["planet"].y) ** 2 +
                                             (points[:, 2] - self._sim_instance.particles["planet"].z) ** 2)
                mask = (los_dist_to_planet < self._sim_instance.particles["planet"].r) & \
                       (points[:, 0] - self._sim_instance.particles["planet"].x < 0)
            else:
                mask = np.ones(len(points), dtype=bool)
            dtfe_obj = DTFE.DTFE(points[mask, 1:3], velocities[mask, 1:3], phys_weights[mask])
        elif d == 3:
            dtfe_obj = DTFE3D.DTFE(points, velocities, phys_weights)
        else:
            raise ValueError("Invalid dimension in DTFE.")

        if grid:
            if d == 2:
                X, Y = self.__grid(timestep)
                dens = dtfe_obj.density(X.flat, Y.flat).reshape((300, 300)) / 1e4
            elif d == 3:
                X, Y, Z = self.__grid(timestep, plane='3d')
                dens = dtfe_obj.density(X.flat, Y.flat, Z.flat).reshape((100, 100, 100)) / 1e6
        else:
            if d == 2:
                dens = dtfe_obj.density(points[:, 0], points[:, 1]) / 1e4
            elif d == 3:
                dens = dtfe_obj.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6

        dens[dens < 0] = 0

        return dens, dtfe_obj.delaunay

    @ensure_data_loaded
    def get_statevectors(self, timestep):
        return self._p_positions, self._p_velocities

    def get_densities(self, timestep, d=3, species_num=1):
        species = self.params.get_species(num=species_num)
        dens, _ = self.dtfe(timestep, species, d=d, grid=False)
        return dens

    # PLOTTING
    def top_down(self, timestep, d=3, colormesh=True, scatter=False, triplot=True, show=True, **kwargs):
        # TOP DOWN DENSITIES
        # ====================================
        kw = {
            "lvlmin": None,
            "lvlmax": None,
            "lim": 10,
            "celest_colors": ['royalblue', 'sandybrown', 'yellow'],
            "smoothing": 1,
            "trialpha": .8,
            "colormap": cm.afmhot,
            "single_plot": False,
            "fill_contour": True,
            "contour": True,
            "show_planet": True,
            "show_moon": True
        }
        kw.update(kwargs)

        ts_list = [timestep] if isinstance(timestep, (int, float)) else timestep

        normalized = False
        for ts in ts_list:
            ts = min([eval(i) for i in list(self.hash_supdict.keys())], key=lambda x: abs(ts - x))
            self.pull_data(ts)
            vis = Visualize(self._sim_instance, lim=kw["lim"], cmap=kw["colormap"], singlePlot=kw["single_plot"])

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)
                points = self._p_positions[np.where(self._p_species == species.id)]

                if len(points) == 0:
                    vis.empty(k)
                    continue

                dens, delaunay = self.dtfe(ts, species, d=d, grid=False)

                if not normalized:
                    if kw['lvlmin'] == 'auto':
                        kw['lvlmin'] = np.log10(np.min(dens[dens > 0])) - .5
                    if kw['lvlmax'] == 'auto':
                        kw['lvlmax'] = np.log10(np.max(dens[dens > 0])) + .5
                    normalized = True

                if colormesh:
                    if d == 3:
                        print("WARNING: Colormesh activated with dim 3. Calculating with dim 2 as this is the only option.")
                    dens_grid, _ = self.dtfe(ts, species, d=2, grid=True)
                    X, Y = self.__grid(ts)
                    vis.add_colormesh(k, X, Y, dens_grid, contour=kw["contour"], fill_contour=kw["fill_contour"], zorder=5, numlvls=25,
                                      celest_colors=kw["celest_colors"], lvlmax=kw['lvlmax'], lvlmin=kw['lvlmin'],
                                      cfilter_coeff=kw["smoothing"], show_planet=kw["show_planet"], show_moon=kw["show_moon"])

                if scatter:
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, perspective="topdown",
                                           cb_format='%.2f', zorder=7, celest_colors=kw["celest_colors"],
                                           vmin=kw["lvlmin"], vmax=kw["lvlmax"], show_planet=kw["show_planet"], show_moon=kw["show_moon"])

                if triplot:
                    if d == 3:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices[:,:3], perspective="topdown",
                                        alpha=kw["trialpha"], celest_colors=kw["celest_colors"],
                                        show_planet=kw["show_planet"], show_moon=kw["show_moon"])
                    elif d == 2:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices, perspective="topdown",
                                        zorder=2, alpha=kw["trialpha"], celest_colors=kw["celest_colors"],
                                        show_planet=kw["show_planet"], show_moon=kw["show_moon"])

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

    #PLOTTING
    def los(self, timestep, show=False, colormesh=True, scatter=False, **kwargs):
        # LINE OF SIGHT DENSITIES
        # ====================================
        kw = {
            "lvlmin": None,
            "lvlmax": None,
            "show_planet": True,
            "show_moon": True,
            "lim": 10,
            "celest_colors": ['royalblue', 'sandybrown', 'yellow'],
            "colormap": cm.afmhot
        }
        kw.update(kwargs)

        ts_list = []
        if isinstance(timestep, int):
            ts_list.append(timestep)
        elif isinstance(timestep, list):
            ts_list = timestep
        elif isinstance(timestep, np.ndarray):
            ts_list = np.ndarray.tolist(timestep)
        else:
            raise TypeError("LOS timestep has an invalid type.")


        running_index = 0
        while running_index < len(ts_list):
            ts = ts_list[::-1][running_index]
            self.pull_data(ts)

            vis = Visualize(self._sim_instance, lim=kw["lim"], cmap=kw["colormap"])

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)

                if colormesh:
                    dens, delaunay = self.dtfe(ts, species, d=2, grid=True, los=True)

                    if running_index == 0:
                        if kw['lvlmin'] == 'auto':
                            kw['lvlmin'] = np.log10(np.min(dens[dens > 0])) - .5
                        if kw['lvlmax'] == 'auto':
                            kw['lvlmax'] = np.log10(np.max(dens[dens > 0])) + .5

                    Y, Z = self.__grid(ts, plane='yz')
                    self.pull_data(ts)
                    vis.add_colormesh(k, -Y, Z, dens, contour=True, fill_contour=True, zorder=9, numlvls=25, perspective='los',
                                      lvlmax=kw['lvlmax'], lvlmin=kw['lvlmin'],
                                      show_planet=kw["show_planet"], show_moon=kw["show_moon"],
                                      celest_colors=kw["celest_colors"])

                    #vis.set_title(r"Particle Densities $log_{10} (N[\mathrm{cm}^{-2}])$ around Planetary Body", size=25)

                if scatter:
                    species = self.params.get_species(num=k + 1)
                    points = self._p_positions[np.where(self._p_species == species.id)]
                    dens, delaunay = self.dtfe(ts, species, d=2, grid=False, los=True)

                    if running_index == 0:
                        if kw['lvlmin'] == 'auto':
                            kw['lvlmin'] = np.log10(np.min(dens[dens > 0])) - .5
                        if kw['lvlmax'] == 'auto':
                            kw['lvlmax'] = np.log10(np.max(dens[dens > 0])) + .5

                    los_dist_to_planet = np.sqrt((points[:, 1] - self._sim_instance.particles["planet"].y) ** 2 +
                                                 (points[:, 2] - self._sim_instance.particles['planet'].z) ** 2)
                    mask = (los_dist_to_planet > self._sim_instance.particles["planet"].r) | (points[:, 0] - self._sim_instance.particles["planet"].x > 0)
                    vis.add_densityscatter(k, -points[:, 1][mask], points[:, 2][mask], dens[mask], perspective="los", cb_format='%.2f',
                                           zorder=5, celest_colors=kw["celest_colors"],
                                           show_planet=kw["show_planet"], show_moon=kw["show_moon"],
                                           vmin=kw["lvlmin"], vmax=kw["lvlmax"])

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

    #PLOTTING
    @ensure_data_loaded
    def plot3d(self, timestep, species_num=1, log_cutoff=None):

        species = self.params.get_species(num=species_num)
        pos = self._p_positions[np.where(self._p_species == species.id)]
        dens, _ = self.dtfe(timestep, species, d=3, grid=False)

        phi, theta = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
        x = self._sim_instance.particles["planet"].r * np.sin(theta) * np.cos(phi) + self._sim_instance.particles["planet"].x
        y = self._sim_instance.particles["planet"].r * np.sin(theta) * np.sin(phi) + self._sim_instance.particles["planet"].y
        z = self._sim_instance.particles["planet"].r * np.cos(theta) + self._sim_instance.particles["planet"].z

        np.seterr(divide='ignore')

        if log_cutoff is not None:
            df = pd.DataFrame({
                'x': pos[:, 0][np.log10(dens) > log_cutoff],
                'y': pos[:, 1][np.log10(dens) > log_cutoff],
                'z': pos[:, 2][np.log10(dens) > log_cutoff]
            })
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.log10(dens[np.log10(dens) > log_cutoff]), opacity=.3)
        else:
            df = pd.DataFrame({
                'x': pos[:, 0],
                'y': pos[:, 1],
                'z': pos[:, 2]
            })
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.log10(dens), opacity=.3)

        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=np.zeros(shape=x.shape), showscale=False))
        fig.update_coloraxes(colorbar_exponentformat='e')
        fig.update_layout(scene_aspectmode='data')
        fig.show()

        np.seterr(divide='warn')

    def logDensities(self, timestep, species_num=1, species_name=None, species_id=None):

        if species_name is not None:
            print(f'Trying to pick species {species_name}')
            species = self.params.get_species(name=species_name)
        elif species_id is not None:
            species = self.params.get_species(id=species_id)
        else:
            species = self.params.get_species(num=species_num)

        dens_max = []
        dens_mean = []
        los_max = []
        los_mean = []

        ts_list = []
        if isinstance(timestep, int):
            ts_list.append(timestep)
        elif isinstance(timestep, list):
            ts_list = [int(x) for x in timestep]
        elif isinstance(timestep, np.ndarray):
            ts_list = np.ndarray.tolist(timestep.astype(int))
        else:
            raise TypeError("top-down timestep has an invalid type. Use 'int', 'list' or 'ndarray'")

        for ts in ts_list:
            print(f"Density calculation at timestep {ts} ...")
            self.pull_data(ts)
            self._sim_instance = self.sa[ts]

            dens2d, _ = self.dtfe(ts, species, d=2, grid=False, los=True)
            dens3d, _ = self.dtfe(ts, species, d=3, grid=False)

            logDens2dInvSort = np.log10(np.sort(dens2d[dens2d > 0])[::-1])
            logDens3dInvSort = np.log10(np.sort(dens3d[dens3d > 0])[::-1])

            while True:
                if logDens2dInvSort[0] > (logDens2dInvSort[20] + 8):
                    logDens2dInvSort = logDens2dInvSort[1:]
                else:
                    break
            while True:
                if logDens3dInvSort[0] > (logDens3dInvSort[20] + 8):
                    logDens3dInvSort = logDens3dInvSort[1:]
                else:
                    break

            logDens3dInvSort[logDens3dInvSort < 0] = 0
            logDens2dInvSort[logDens2dInvSort < 0] = 0

            dens_max.append(logDens3dInvSort[0])
            #dens_mean.append(np.mean(np.array_split(logDens3dInvSort, 2)[0]))
            dens_mean.append(np.mean(logDens3dInvSort))
            los_max.append(logDens2dInvSort[0])
            #los_mean.append(np.mean(np.array_split(logDens2dInvSort, 2)[0]))
            los_mean.append(np.mean(logDens2dInvSort))

        return dens_max, dens_mean, los_max, los_mean

    def transit_curve(self, phase_curve_datapath):
        import matplotlib.pyplot as plt

        first_instance = self.sa[0]

        df = pd.read_pickle(f"./output/phaseCurves/data/PhaseCurveData-{phase_curve_datapath[11:]}.pkl")
        exomoon_phase = df["phases"].values
        exomoon_los_mean = df["mean_los"].values

        exoplanet_orbit = first_instance.particles["planet"].calculate_orbit(primary=first_instance.particles["star"])
        exomoon_orbit = first_instance.particles["moon"].calculate_orbit(primary=first_instance.particles["planet"])
        #data_length = np.around(len(exomoon_phase) * exoplanet_orbit.P / exomoon_orbit.P).astype(int)

        delta_t = 5 * self.params.int_spec["sim_advance"] * exomoon_orbit.P
        exoplanet_jump = 360/exoplanet_orbit.P * delta_t
        exoplanet_phases = np.arange(-180, 180, exoplanet_jump)
        data_length = len(exoplanet_phases)

        # Exomoon:
        exomoon_transit_depth_array = []
        exomoon_los_mean = np.concatenate((np.array_split(exomoon_los_mean, 2)[::-1]))
        exomoon_los_mean_array = np.array_split(np.resize(exomoon_los_mean, 6*data_length), 6)
        for i in range(len(exomoon_los_mean_array)):
            exomoon_transit_depth_array.append(np.exp(-10**exomoon_los_mean_array[i] * 9.84e-13))

        # Exoplanet:
        exoplanet_shadow_phase = np.arctan2(first_instance.particles["star"].r,
                                            first_instance.particles["planet"].calculate_orbit(
                                                primary=first_instance.particles["star"]).a)
        transit_ingress_phase = -exoplanet_shadow_phase * 180 / np.pi
        transit_egress_phase = exoplanet_shadow_phase * 180 / np.pi
        exoplanet_star_radius_ratio = first_instance.particles["planet"].r / first_instance.particles["star"].r
        exoplanet_transit_depth1 = np.ones(data_length)
        exoplanet_transit_depth1[np.where((exoplanet_phases > transit_ingress_phase) & (exoplanet_phases < transit_egress_phase))] = 1 - exoplanet_star_radius_ratio**2

        temp_p, temp_s = 963, 4715
        exoplanet_transit_depth2 = np.ones(data_length)
        exoplanet_transit_depth2[np.where((exoplanet_phases > transit_ingress_phase) & (exoplanet_phases < transit_egress_phase))] = 1 - exoplanet_star_radius_ratio ** 2 * temp_p/temp_s

        fig = plt.figure(figsize=(12,3), dpi=100)
        #gs = fig.add_gridspec(2, 6, wspace=0.01, hspace=0.01)
        gs = fig.add_gridspec(1, 6, wspace=0.01)
        axs = gs.subplots()

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False, zorder=10)
        for i in range(6):
            if i % 2 == 0:
                axs[i].plot(exoplanet_phases, exoplanet_transit_depth1, label='Exoplanet', color='k')
                axs[i].plot(exoplanet_phases, exomoon_transit_depth_array[i], label='Exomoon', color='orange')
                axs[i].set_xlim(-50, 50)
                axs[i].set_ylim(0.98, 1.001)
                axs[i].set_title("Transit", fontsize=12)

                #axs[1][i].plot(exoplanet_phases, exoplanet_transit_depth1, color='k')
                #axs[1][i].set_xlim(-50, 50)
                #axs[1][i].set_ylim(0.978, 0.985)
            else:
                axs[i].plot(exoplanet_phases+180, exoplanet_transit_depth2, label='Exoplanet', color='k')
                axs[i].plot(exoplanet_phases+180, exomoon_transit_depth_array[i], label='Exomoon', color='orange')
                axs[i].set_xlim(-50+180, 50+180)
                axs[i].set_ylim(0.98, 1.001)
                axs[i].set_title("Occultation", fontsize=12)

                #axs[1][i].plot(exoplanet_phases, exoplanet_transit_depth2, label='Exoplanet', color='k')
                #axs[1][i].plot(exoplanet_phases, exomoon_transit_depth_array[i], label='Exomoon', color='orange')
                #axs[1][i].set_xlim(-50, 50)
                #axs[1][i].set_ylim(0.978, 0.985)

            axs[i].spines.bottom.set_visible(True)
            #axs[1][i].spines['top'].set_linestyle('dashed')
            #axs[1][i].spines['top'].set_capstyle("butt")

            if i > 0:
                axs[i].get_yaxis().set_visible(False)
                axs[i].spines['left'].set_linestyle('dashed')
                axs[i].spines['left'].set_capstyle("butt")
                #axs[1][i].get_yaxis().set_visible(False)
                #axs[1][i].spines['left'].set_linestyle('dashed')
                #axs[1][i].spines['left'].set_capstyle("butt")
            if i < 5:
                axs[i].spines.right.set_visible(False)
                #axs[1][i].spines.right.set_visible(False)
                #axs[1][i].plot([1.01, 1], [0, 2], transform=axs[1][i].transAxes, **kwargs)


            axs[i].locator_params(axis='x', nbins=5)
            #axs[1][i].locator_params(axis='x', nbins=5)
            #axs[1][i].zorder = 5 - i
            axs[i].tick_params(axis='x', which='major', labelsize=10, pad=5)

        axs[-1].legend(loc='lower center', fontsize=12, framealpha=0)

        axs[0].tick_params(axis='y', which='major', labelsize=15)
        #axs[1][0].tick_params(axis='y', which='major', labelsize=10)

        fig.text(0.09, 0.043, r'$\phi_p$ [$^{\circ}$]:', ha='left', fontsize=10)

        #plt.savefig("Transit-W69-physical.png")
        plt.show()


class PhaseCurve(SerpensAnalyzer):

    def __init__(self, title='unnamed', archive_from=None):
        self.title = title

        if archive_from is not None:
            if "simulation-" not in archive_from:
                print("Error, wrong format in archive_from argument.")
                print("Please enter a 'simulation-' folder name that has a previously run serpens simulation archive inside.")
                print("Example: 'simulation-W49-ExoIo-Na-3h-HV' \n")
                raise Exception()

            print("\t copying ...")
            shutil.copy2(f'{os.getcwd()}/schedule_archive/{archive_from}/archive.bin', f'{os.getcwd()}')
            shutil.copy2(f'{os.getcwd()}/schedule_archive/{archive_from}/hash_library.pickle', f'{os.getcwd()}')
            shutil.copy2(f'{os.getcwd()}/schedule_archive/{archive_from}/Parameters.pickle', f'{os.getcwd()}')
            shutil.copy2(f'{os.getcwd()}/schedule_archive/{archive_from}/Parameters.txt', f'{os.getcwd()}')
            print("\t ... done!")

        super().__init__(reference_system="geocentric")

    def calculate_curve(self, save_data=True):

        print("Calculating phase curve for present simulation instance... ")

        second_instance_index = int(list(self.hash_supdict.keys())[1])
        step = second_instance_index - 1
        second_instance = self.sa[second_instance_index]
        orbit_phase = np.around(second_instance.particles["moon"].calculate_orbit(primary=second_instance.particles["planet"]).theta * 180 / np.pi)
        orbit_first_index = len(self.sa) - (second_instance_index + 360 / orbit_phase * second_instance_index)

        if (orbit_first_index - orbit_first_index % step + 1) > second_instance_index:
            ts_list = np.arange(orbit_first_index - orbit_first_index % step + 1,
                                len(self.sa) - len(self.sa) % step + 1, step)
        else:
            ts_list = np.arange(orbit_first_index - orbit_first_index % step + 1,
                                len(self.sa) - len(self.sa) % step + 1, step)

        first_instance = self.sa[0]
        exomoon_orbit = first_instance.particles["moon"].calculate_orbit(primary=first_instance.particles["planet"])
        exoplanet_orbit = first_instance.particles["planet"].calculate_orbit(primary=first_instance.particles["star"])

        phases = np.zeros(len(ts_list))

        #dl = np.zeros(len(ts_list))
        for i, timestep in enumerate(ts_list):
            self.pull_data(int(timestep))
            phase = self._sim_instance.particles["moon"].calculate_orbit(
                primary=self._sim_instance.particles["planet"]).theta * 180 / np.pi
            if self.reference_system == 'geocentric':
                shift = (timestep * self.params.int_spec["sim_advance"] * exomoon_orbit.P / exoplanet_orbit.P) * 360
                phases[i] = (phase + shift) % 360
            else:
                phases[i] = phase

            #p, v = self.get_statevectors(timestep)
            #vx = v[:,0][3:]
            #dl[i] = np.mean(-vx/299792458)

        #print(f"Maximum doppler: {np.max(np.abs(dl))}")
        #return

        sort_index = np.argsort(phases)
        phases = np.asarray(phases)[sort_index]
        ts_list = ts_list[sort_index]

        dens_max, dens_mean, los_max, los_mean = self.logDensities(ts_list)

        if save_data:
            d = {"phases": phases, "timestep": ts_list, "max_dens": dens_max, "max_los": los_max,
                 "mean_dens": dens_mean, "mean_los": los_mean}
            df = pd.DataFrame(data=d)

            if not os.path.exists(f'output/phaseCurves/data'):
                os.makedirs(f'output/phaseCurves/data')

            df.to_pickle(f"./output/phaseCurves/data/PhaseCurveData-{self.title}.pkl")

    def plot_curve_local(self):
        print("Local not yet implemented")
        pass

    @staticmethod
    def plot_curve_external(exoplanet_system, lifetime='photo', title='unnamed',
                            savefig=False, column_dens=True, part_dens=False):
        import json


        def hex_to_RGB(hex_str):
            """ #FFFFFF -> [255,255,255]"""
            # Pass 16 to the integer function for change of base
            return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]

        def get_color_gradient(c1, c2, n):
            """
            Given two hex colors, returns a color gradient
            with n colors.
            """
            assert n > 1
            c1_rgb = np.array(hex_to_RGB(c1)) / 255
            c2_rgb = np.array(hex_to_RGB(c2)) / 255
            mix_pcts = [x / (n - 1) for x in range(n)]
            rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
            return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]

        def lighten_color(color, amount=0.5):
            """
            Lightens the given color by multiplying (1-luminosity) by the given amount.
            Input can be matplotlib color string, hex string, or RGB tuple.

            Examples:
            >> lighten_color('g', 0.3)
            >> lighten_color('#F034A3', 0.6)
            >> lighten_color((.3,.55,.1), 0.5)
            """
            import matplotlib.colors as mc
            import colorsys
            try:
                c = mc.cnames[color]
            except:
                c = color
            c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

        if lifetime == 'photo':
            tau = 'physical'
        elif lifetime == '3h':
            tau = '3h'
        else:
            print("Please set 'lifetime' to either be 'photo' or '3h' to continue.")
            return

        # Set up link of exoplanet to data file
        implemented = {'WASP-49': 'W49',
                       'HD-189733': 'HD189',
                       'HD-209458': 'HD209',
                       'HAT-P-1': 'HATP1',
                       'WASP-39': 'W39',
                       'WASP-17': 'W17',
                       'WASP-69': 'W69',
                       'WASP-96': 'W96',
                       'XO-2N': 'XO2N',
                       'WASP-52': 'W52'}

        # Get celestial object parameters
        with open('resources/objects.txt') as f:
            data = f.read().splitlines(True)
            obj = json.loads(data[[f"{exoplanet_system}" in s for s in data].index(True)])
        star_r = obj["star"]["r"]
        star_m = obj["star"]["m"]
        planet_a = obj["planet"]["a"]
        planet_r = obj["planet"]["r"]
        planet_m = obj["planet"]["m"]
        moon_a = obj["moon"]["a"]
        moon_r = obj["moon"]["r"]

        # Get inclination
        if "inc" in obj["planet"]:
            planet_inc = obj["planet"]["inc"]
        else:
            planet_inc = 0

        # Calculate orbital periods
        G = 6.6743e-11
        planet_P = 2 * np.pi * np.sqrt(planet_a ** 3 / (G * star_m))
        moon_P = 2 * np.pi * np.sqrt(moon_a ** 3 / (G * planet_m))

        # Calculate shadow and LOS blockade
        shadow_phase = np.arctan2(planet_r, moon_a)
        los_ingress_phase = (np.pi - shadow_phase) * 180 / np.pi
        los_egress_phase = (np.pi + shadow_phase) * 180 / np.pi
        shadow_ingress_phase = (2 * np.pi - shadow_phase) * 180 / np.pi
        shadow_egress_phase = shadow_phase * 180 / np.pi

        # Calculate transit duration
        impact_parameter = 0
        radius_ratio = planet_r / star_r
        transit_duration_total = planet_P / np.pi * \
                                 np.arcsin(star_r / planet_a *
                                           np.sqrt((1 + radius_ratio) ** 2 - impact_parameter ** 2)
                                           / np.sin(np.pi / 2 - planet_inc))
        phases_per_transit = 2 * np.pi / moon_P * transit_duration_total

        # Set up data retrieval paths
        load_paths = [#f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEarth-SO2-3h-HV-WSHAD',
                      #f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEarth-K-HV',
                      #f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEnce-SO2-3h-HV-WSHAD',
                      #f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEnce-K-HV',
                      #f'simulation-{implemented[f"{exoplanet_system}"]}-ExoIo-SO2-3h-HV-WSHAD',
                      f'simulation-{implemented[f"{exoplanet_system}"]}-ExoIo-Na-physical-HV-NOSHAD',
            # f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEarth-Na-{tau}-HV-NOSHAD',
            # f'simulation-{implemented[f"{exoplanet_system}"]}-ExoEnce-Na-{tau}-HV-NOSHAD',
            # f'simulation-{implemented[f"{exoplanet_system}"]}-ExoIo-Na-{tau}-HV-NOSHAD',
        ]
        # Retrieve data
        phase_suparray = []
        los_mean_suparray = []
        dens_mean_suparray = []
        for path in load_paths:
            df = pd.read_pickle(f"./output/phaseCurves/data/PhaseCurveData-{path[11:]}.pkl")
            #df = pd.read_pickle(f"./output/phaseCurves/data/PhaseCurveData-W69-ExoIo-Na-physical-HV-NORAD.pkl")
            #df = pd.read_pickle(f"./output/phaseCurves/data/PhaseCurveData-HD209-ExoIo-Na-physical-HV-NORAD.pkl")
            phase_values = df["phases"].values
            phase_suparray.append(phase_values)
            los_mean_suparray.append(df["mean_los"].values)
            dens_mean_suparray.append(df["mean_dens"].values)
            #break

        # Colors for background color gradient
        color1 = matplotlib.colors.to_hex('ivory')
        color2 = matplotlib.colors.to_hex('darkorange')

        # Number of subplots
        rows = 0
        if column_dens:
            rows += 1
        if part_dens:
            rows += 1

        # Plot figure
        fig = plt.figure(figsize=(12, 6 * rows), dpi=100)
        axs = []
        axs_index = 0
        for i in range(rows):
            ax = fig.add_subplot(rows, 1, i + 1)
            axs.append(ax)
            transit_arrow = FancyArrowPatch(posA=(45 / 360, .1),
                                            posB=((45 + phases_per_transit * 180 / np.pi) / 360, .1),
                                            arrowstyle='|-|', color='k',
                                            shrinkA=0, shrinkB=0, mutation_scale=10, zorder=10,
                                            transform=axs[axs_index].transAxes,
                                            linewidth=1.5)
            axs[axs_index].text((45 + (phases_per_transit * 180 / np.pi) / 2) / 360, 0.11, 'Transit',
                                ha='center', transform=axs[axs_index].transAxes,
                                fontsize=20, zorder=5)
            axs[axs_index].add_artist(transit_arrow)

            if '3h' in load_paths[0]:
                axs[axs_index].text(.25, .9, r'$\tau = 3\mathrm{h}$',
                                    ha='center', transform=axs[axs_index].transAxes,
                                    fontsize=27, color='r', zorder=5)
            else:
                axs[axs_index].text(.25, .9, r'$\tau = \tau_\gamma$',
                                    ha='center', transform=axs[axs_index].transAxes,
                                    fontsize=27, color='r', zorder=5)

        if part_dens:
            for ind in range(len(load_paths)):
                sig = load_paths[ind][11:][(load_paths[ind][11:].index('-') + 4):(load_paths[ind][11:].index('-') + 6)]
                if sig == "Ea":
                    label = r"$(M,R)_\oplus$"
                    color = 'springgreen'
                elif sig == "Io":
                    label = r"$(M,R)_\mathrm{Io}$"
                    color = "orangered"
                elif sig == "En":
                    label = r"$(M,R)_\mathrm{Enc}$"
                    color = 'deepskyblue'
                else:
                    label = ''
                    color = 'k'

                step_size = np.ediff1d(phase_suparray[ind])
                phases = []
                dens_mean = []
                for p in range(len(phase_suparray[ind]) - 1):
                    if np.abs(phase_suparray[ind][p + 1] - phase_suparray[ind][p]) >= 1/2*np.mean(step_size):
                        phases.append(phase_suparray[ind][p])
                        dens_mean.append(dens_mean_suparray[ind][p])

                # Smooth the curve and plot
                spl = make_interp_spline(phases, dens_mean, k=3)
                X_ = np.linspace(min(phases), max(phases), 200)
                Y_ = spl(X_)
                axs[axs_index].plot(X_, Y_, label=label, color=color, linewidth=3, alpha=.9, zorder=3)

                if 'WSHAD' not in load_paths[ind]:
                    print(f"Particle density orbit-average ({sig}): {np.mean(Y_)}, +{np.max(Y_) - np.mean(Y_)}, -{np.mean(Y_) - np.min(Y_)}")

                #window = 1
                #while True:
                #   if X_[window] - X_[0] < phases_per_transit * 180/np.pi:
                #       window += 1
                #   else:
                #       break
                #conv = np.convolve(np.resize(Y_, 2*len(Y_) - (len(X_) - window + 1)), np.ones(window)/window, mode='valid')
                #axs[axs_index].plot(X_, conv, label=label, color=color, linewidth=3, alpha=.9)

                #from scipy.optimize import curve_fit
                #
                #def form(phi, I_0, w, phi0, offset):
                #    return I_0 * np.cos(w * np.radians(phi - phi0)) + offset
                #
                #if sig == "Io":
                #    param, covariance = curve_fit(form, X_, Y_, method='trf')
                #    print(*param)
                #    axs[axs_index].plot(X_, form(X_, *param), label='analytical '+r'($\mathrm{Io}$)', zorder=2, alpha=.8, c='orangered', linestyle='--')
                #
                #break

            axs[axs_index].set_ylabel(r"log $\bar{n}$ [cm$^{-3}$]", fontsize=18)
            axs[axs_index].set_xlim(left=0, right=360)
            axs[axs_index].tick_params(axis='both', which='major', labelsize=19)
            if rows == 1:
                axs[axs_index].set_xlabel(r"exomoon phase $\phi \in$ [0, 360]$^\circ$", fontsize=18)
            axs_index += 1

        temp_cache = 0
        if column_dens:
            for ind in range(len(load_paths)):
                sig = load_paths[ind][11:][(load_paths[ind][11:].index('-') + 4):(load_paths[ind][11:].index('-') + 6)]
                if sig == "Ea":
                    label = r"$(M,R)_\oplus$"
                    color = 'navy'
                elif sig == "Io":
                    label = r"$(M,R)_\mathrm{Io}$"
                    color = 'orangered'
                elif sig == "En":
                    label = r"$(M,R)_\mathrm{Enc}$"
                    color = 'teal'
                else:
                    label = ''
                    color = 'k'

                step_size = np.ediff1d(phase_suparray[ind])
                phases = []
                los_mean = []
                for p in range(len(phase_suparray[ind]) - 1):
                    if np.abs(phase_suparray[ind][p + 1] - phase_suparray[ind][p]) >= 1/2*np.mean(step_size):
                        phases.append(phase_suparray[ind][p])
                        los_mean.append(los_mean_suparray[ind][p])
                spl = make_interp_spline(phases, los_mean, k=3)
                X_ = np.linspace(min(phases), max(phases), 200)
                Y_ = spl(X_)

                #print(f"Column density orbit-average ({sig}): {np.mean(Y_)}, +{np.max(Y_) - np.mean(Y_)}, -{np.mean(Y_) - np.min(Y_)}")

                if 'WSHAD' not in load_paths[ind]:
                    print(f"Column density orbit-average ({sig}):")
                    print('\t' + f"{np.mean(Y_)}, +{np.max(Y_) - np.mean(Y_)}, -{np.mean(Y_) - np.min(Y_)}")
                    print('\t' + f"delta: {np.min(Y_) - temp_cache}")
                    transit_depth = 1 - np.exp(-10 ** Y_ * 9.84e-13)  #
                    print('\t' + f"Transit depth: {np.mean(transit_depth)}, "
                         f"+{np.max(transit_depth) - np.mean(transit_depth)}, "
                         f"-{np.mean(transit_depth) - np.min(transit_depth)}")

                else:
                    #print(f"Column density orbit-average ({sig}, WSHAD):")
                    #print('\t' + f"{np.mean(Y_)}, +{np.max(Y_) - np.mean(Y_)}, -{np.mean(Y_) - np.min(Y_)}")
                    temp_cache = np.min(Y_)

                #print(f"ATD ({sig}): {len(Y_[Y_ > 8]) / len(Y_) * 100} %")
                #print(f"ATP ({sig}): {np.max(Y_) - 8}")

                axs[axs_index].plot(X_, Y_, label=label, color=color, linewidth=3, alpha=.9, zorder=3)

                #window = 1
                #while True:
                #    if X_[window] - X_[0] < phases_per_transit * 180/np.pi:
                #        window += 1
                #    else:
                #        break
                #conv = np.convolve(np.resize(Y_, 2*len(Y_) - (len(X_) - window + 1)), np.ones(window)/window, mode='valid')
                #axs[axs_index].plot(X_, conv, label=label, color=color, linewidth=3, alpha=.9)
                #
                #transit_depth = np.exp(-10 ** conv * 9.84e-13)
                #print(f"Transit depth ({sig}): {np.mean(transit_depth)}, "
                #      f"+{np.max(transit_depth) - np.mean(transit_depth)}, "
                #      f"-{np.mean(transit_depth) - np.min(transit_depth)}"
                #      )
                #
                #from scipy.optimize import curve_fit
                #
                #def form2(phi, I_0, I_1, phi0, offset):
                #    return I_0 * np.cos(np.radians(phi - phi0)) + I_1 * np.cos(2 * np.radians(phi - phi0)) + offset
                #
                #if sig == "Io":
                #    param, covariance = curve_fit(form2, X_, Y_, method='trf')
                #    print(*param)
                #    axs[axs_index].plot(X_, form2(X_, *param), label='analytical '+r'($\mathrm{Io}$)', zorder=2, alpha=.8, c='orangered', linestyle='--')
                #
                #break

            axs[axs_index].set_ylabel(r"log $\bar{N}$ [cm$^{-2}$]", fontsize=20)
            axs[axs_index].set_xlim(left=0, right=360)
            axs[axs_index].set_xlabel(r"exomoon phase $\phi \in$ [0, 360]$^\circ$", fontsize=20)
            axs[axs_index].tick_params(axis='both', which='major', labelsize=22)

            axs[axs_index].axvline(los_ingress_phase, linestyle='--', color='gray')
            axs[axs_index].axvline(los_egress_phase, linestyle='--', color='gray')
            # axs[k].axvspan(los_ingress_phase, los_egress_phase, facecolor='black', alpha=0.5,
            #               hatch='/', fill=False, label="_" * i + "Hidden")

        if part_dens:
            axs[0].set_title(f"Phase-Density Curves: $\mathbf{{{title}}}$", fontsize=22)
        else:
            axs[0].set_title(f"Phase-Density Curves: $\mathbf{{{title}}}$", fontsize=22)

        colors1 = get_color_gradient(color1, color2, 60)
        colors2 = get_color_gradient(color2, color1, 60)
        first_range = np.linspace(shadow_egress_phase, 180, 60)
        second_range = np.linspace(180, shadow_ingress_phase, 60)

        for i in range(0, 59):
            for k, ax in enumerate(axs):
                ax.axvspan(first_range[i], first_range[i + 1], facecolor=colors2[i], alpha=0.8)
                ax.axvspan(second_range[i], second_range[i + 1], facecolor=colors1[i], alpha=0.8)

        colors_shadow = get_color_gradient(matplotlib.colors.to_hex('black'), matplotlib.colors.to_hex('saddlebrown'),
                                           20)
        first_shadow_range = np.linspace(0, shadow_egress_phase, 20)
        second_shadow_range = np.linspace(shadow_ingress_phase, 360, 20)
        for i in range(0, 19):
            for k, ax in enumerate(axs):
                ax.axvspan(second_shadow_range[i], second_shadow_range[i + 1], facecolor=np.flip(colors_shadow)[i],
                           alpha=1,
                           label="_" * abs(i - 8) + "Shadow")
                ax.axvspan(first_shadow_range[i], first_shadow_range[i + 1], facecolor=colors_shadow[i], alpha=1)

        for k, ax in enumerate(axs):
            ax.locator_params(axis='y', nbins=7, tight=True)
            #leg = ax.legend(loc=(.65, .05), framealpha=1, fontsize=19, frameon=True)
            #for lh in leg.legendHandles:
            #    lh.set_alpha(1)

        plt.tight_layout()

        if not os.path.exists(f'output/phaseCurves'):
            os.makedirs(f'output/phaseCurves')

        if savefig:
            plt.savefig(f'output/phaseCurves/{title}.png', dpi=100, bbox_inches='tight')
        else:
            plt.show()