import numpy as np
import os as os
import glob
import shutil
import rebound
import reboundx
import dill
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src import DTFE, DTFE3D
from parameters import Parameters
from src.visualize import Visualize


class SerpensAnalyzer:

    def __init__(self, save_output=False, save_archive=False, folder_name=None, z_cutoff=None, r_cutoff=None, reference_system="heliocentric"):
        # PICKLE:
        # ===================
        # print("WARNING: SERPENS is about to unpickle particle data.
        # Pickle files are not secure. Make sure you trust the source!")
        # input("\t Press Enter to continue...")

        self.hash_supdict = {}
        with open('hash_library.pickle', 'rb') as f:
            while True:
                try:
                    a = dill.load(f)
                    dict_timestep = list(a.keys())[0]
                except EOFError:
                    break
                else:
                    self.hash_supdict[dict_timestep] = a[dict_timestep]

        try:
            #with open('hash_library.pickle', 'rb') as handle:
            #    self.hash_supdict = dill.load(handle)

            with open('Parameters.pickle', 'rb') as handle:
                params_load = dill.load(handle)
                params_load()
        except Exception:
            raise Exception("hash_library.pickle and/or Parameters.pickle not found.")

        try:
            self.sa = rebound.SimulationArchive("archive.bin", process_warnings=False)
        except Exception:
            raise Exception("simulation archive not found.")

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

        self.z_cutoff = z_cutoff
        self.r_cutoff = r_cutoff
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

    def __grid(self, timestep, plane='xy'):
        self.__pull_data(timestep)
        #sim_instance = self.sa[timestep]

        if self.moon_exists:
            boundary = self.params.int_spec["r_max"] * self._sim_instance.particles["moon"].calculate_orbit(
                primary=self._sim_instance.particles["planet"]).a

            if plane == 'xy':
                offsetx = self._sim_instance.particles["planet"].x
                offsety = self._sim_instance.particles["planet"].y
                offsetz = 0
            elif plane == 'yz':
                offsetx = self._sim_instance.particles["planet"].y
                offsety = self._sim_instance.particles["planet"].z
                offsetz = 0
            elif plane == '3d':
                offsetx = self._sim_instance.particles["planet"].x
                offsety = self._sim_instance.particles["planet"].y
                offsetz = self._sim_instance.particles["planet"].z
            else:
                raise ValueError("Invalid plane in grid construction!")

        else:
            boundary = self.params.int_spec["r_max"] * self._sim_instance.particles["planet"].a
            offsetx = 0
            offsety = 0
            offsetz = 0

        if not plane == '3d':
            X, Y = np.meshgrid(np.linspace(-boundary + offsetx, boundary + offsetx, 100),
                               np.linspace(-boundary + offsety, boundary + offsety, 100))
            return X, Y
        else:
            X, Y, Z = np.meshgrid(np.linspace(-boundary + offsetx, boundary + offsetx, 100),
                                  np.linspace(-boundary + offsety, boundary + offsety, 100),
                                  np.linspace(-boundary + offsetz, boundary + offsetz, 100))
            return X, Y, Z

    def __pull_data(self, timestep):

        if self.cached_timestep == timestep:
            return
        else:
            self.cached_timestep = timestep

        # REBX: sim_instance, rebx = self.sa[timestep]
        self._sim_instance = self.sa[timestep]
        self._p_positions = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._p_velocities = np.zeros((self._sim_instance.N, 3), dtype="float64")
        self._p_hashes = np.zeros(self._sim_instance.N, dtype="uint32")
        self._p_species = np.zeros(self._sim_instance.N, dtype="int")
        self._p_weights = np.zeros(self._sim_instance.N, dtype="float64")
        self._sim_instance.serialize_particle_data(xyz=self._p_positions, vxvyvz=self._p_velocities,
                                                   hash=self._p_hashes)

        if self.reference_system == "geocentric":
            phase = np.arctan2(self._sim_instance.particles["planet"].y, self._sim_instance.particles["planet"].x)
            c, s = np.cos(phase), np.sin(phase)
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            self._p_positions = (R @ self._p_positions.T).T
            self._p_velocities = (R @ self._p_velocities.T).T

            reb_rot = rebound.Rotation(angle=phase, axis='z')
            self._sim_instance.particles[2].rotate(reb_rot.inverse())
            self._sim_instance.particles[1].rotate(reb_rot.inverse())
            self._sim_instance.particles[0].rotate(reb_rot.inverse())

        if not timestep == 0:
            hash_dict_current = self.hash_supdict[str(timestep)]
        else:
            hash_dict_current = {}

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

            #particle_iter = hash_dict_current[str(self._p_hashes[k1])]["i"]
            #
            #if self.moon_exists:
            #    particle_time = (timestep - particle_iter) * self.params.int_spec["sim_advance"] * \
            #                    sim_instance.particles[
            #                        "moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
            #else:
            #    particle_time = (timestep - particle_iter) * self.params.int_spec["sim_advance"] * \
            #                    sim_instance.particles[
            #                        "planet"].P
            #
            #if self.params.get_species(id=self._p_species[k1]) is None:
            #    continue
            #
            #chem_network = self.params.get_species(id=self._p_species[k1]).network
            #reaction_rate = 0
            #if not isinstance(chem_network, (int, float)):
            #    for l in range(np.size(chem_network[:, 0])):
            #        reaction_rate += 1 / float(chem_network[:, 0][l])
            #else:
            #    reaction_rate = 1 / chem_network
            #self._p_weights[k1] = np.exp(-particle_time * reaction_rate)

        if self.z_cutoff is not None:
            assert isinstance(self.z_cutoff, (float, int))
            mask = (self._p_positions[:,2] < self.z_cutoff * self._sim_instance.particles["planet"].r) \
                   & (self._p_positions[:,2] > -self.z_cutoff * self._sim_instance.particles["planet"].r)
            self._p_positions = self._p_positions[mask]
            self._p_velocities = self._p_velocities[mask]
            self._p_hashes = self._p_hashes[mask]
            self._p_species = self._p_species[mask]
            self._p_weights = self._p_weights[mask]

        if self.r_cutoff is not None:
            assert isinstance(self.r_cutoff, (float, int))
            r = np.linalg.norm(self._p_positions - self._sim_instance.particles["planet"].xyz, axis=1)
            mask = r < self.r_cutoff * self._sim_instance.particles["planet"].r
            self._p_positions = self._p_positions[mask]
            self._p_velocities = self._p_velocities[mask]
            self._p_hashes = self._p_hashes[mask]
            self._p_species = self._p_species[mask]
            self._p_weights = self._p_weights[mask]

    def dtfe(self, species, timestep, d=2, grid=True, los=False):
        self.__pull_data(timestep)

        if self.moon_exists:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * self._sim_instance.particles["moon"].calculate_orbit(primary=self._sim_instance.particles["planet"]).P
        else:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * self._sim_instance.particles["planet"].P

        points = self._p_positions[np.where(self._p_species == species.id)]
        velocities = self._p_velocities[np.where(self._p_species == species.id)]
        weights = self._p_weights[np.where(self._p_species == species.id)]

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
                los_dist_to_planet = np.sqrt((points[:, 1] - self._sim_instance.particles["planet"].y) ** 2 +
                                             (points[:, 2] - self._sim_instance.particles["planet"].z) ** 2)
                mask = (los_dist_to_planet < self._sim_instance.particles["planet"].r) & (points[:, 0] - self._sim_instance.particles["planet"].x < 0)

                dtfe = DTFE.DTFE(points[:, 1:3], velocities[:, 1:3], phys_weights)
                if grid:
                    Y, Z = self.__grid(timestep, plane='yz')
                    dens = dtfe.density(Y.flat, Z.flat).reshape((100, 100)) / 1e4
                else:
                    dens = dtfe.density(points[:, 1], points[:, 2]) / 1e4
                    dens[mask] = 0

            else:
                dtfe = DTFE.DTFE(points[:, :2], velocities[:, :2], phys_weights)
                if grid:
                    X, Y = self.__grid(timestep)
                    dens = dtfe.density(X.flat, Y.flat).reshape((100, 100)) / 1e4
                else:
                    dens = dtfe.density(points[:, 0], points[:, 1]) / 1e4

        elif d == 3:
            dtfe = DTFE3D.DTFE(points, velocities, phys_weights)
            if not los and grid:
                X, Y, Z = self.__grid(timestep, plane='3d')
                dens = dtfe.density(X.flat, Y.flat, Z.flat).reshape((100, 100, 100)) / 1e6
            else:
                dens = dtfe.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6

        else:
            raise ValueError("Invalid dimension in DTFE.")

        dens[dens < 0] = 0
        return dens, dtfe.delaunay

    def get_statevectors(self, timestep):
        self.__pull_data(timestep)
        return self._p_positions, self._p_velocities

    def get_densities(self, timestep, d=3, species_num=1):
        species = self.params.get_species(num=species_num)
        dens, _ = self.dtfe(species, timestep, d=d, grid=False)
        return dens

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
            "contour": True
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
            raise TypeError("top-down timestep has an invalid type.")

        running_index = 0
        while running_index < len(ts_list):
            ts = ts_list[running_index]
            self.__pull_data(ts)

            vis = Visualize(self._sim_instance, lim=kw["lim"], cmap=kw["colormap"], singlePlot=kw["single_plot"])

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)
                points = self._p_positions[np.where(self._p_species == species.id)]
                dens, delaunay = self.dtfe(species, ts, d=d, grid=False)

                if colormesh:
                    if d == 3:
                        print("WARNING: Colormesh activated with dim 3. Calculating with dim 2 as this is the only option.")
                    dens_grid, _ = self.dtfe(species, ts, d=2, grid=True)
                    X, Y = self.__grid(ts)
                    self.__pull_data(ts)
                    vis.add_colormesh(k, X, Y, dens_grid, contour=kw["contour"], fill_contour=kw["fill_contour"], zorder=2, numlvls=25,
                                      celest_colors=kw["celest_colors"], lvlmax=kw['lvlmax'], lvlmin=kw['lvlmin'], cfilter_coeff=kw["smoothing"])

                if scatter:
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, perspective="topdown",
                                           cb_format='%.2f', zorder=1, celest_colors=kw["celest_colors"],
                                           vmin=kw["lvlmin"], vmax=kw["lvlmax"])

                if triplot:
                    if d == 3:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices[:,:3], perspective="topdown",
                                        alpha=kw["trialpha"], celest_colors=kw["celest_colors"])
                    elif d == 2:
                        vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices, perspective="topdown",
                                        zorder=2, alpha=kw["trialpha"], celest_colors=kw["celest_colors"])

                if d == 2:
                    vis.set_title(r"Particle Densities $log_{10} (N[\mathrm{cm}^{-2}])$ around Planetary Body", size=25)
                elif d == 3:
                    vis.set_title(r"Particle Densities $log_{10} (n[\mathrm{cm}^{-3}])$ around Planetary Body", size=25)

            if self.save:
                vis(show_bool=show, save_path=self.path, filename=f'TD_{ts}_{self.save_index}')
                self.save_index += 1

                # Handle saving bugs...
                list_of_files = glob.glob(f'./output/{self.path}/plots/*')
                latest_file = max(list_of_files, key=os.path.getctime)
                if os.path.getsize(latest_file) < 50000:
                    print("\t Detected low filesize (threshold at 50 KB). Possibly encountered a saving bug. Retrying process.")
                    os.remove(latest_file)
                else:
                    running_index += 1

            else:
                vis(show_bool=show)
                running_index += 1

            del vis

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
            ts = ts_list[running_index]
            self.__pull_data(ts)

            vis = Visualize(self._sim_instance, lim=kw["lim"], cmap=kw["colormap"])

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)

                if colormesh:
                    dens, delaunay = self.dtfe(species, ts, d=2, grid=True, los=True)

                    Y, Z = self.__grid(ts, plane='yz')
                    self.__pull_data(ts)
                    vis.add_colormesh(k, -Y, Z, dens, contour=True, fill_contour=True, zorder=9, numlvls=25, perspective='los',
                                      lvlmax=kw['lvlmax'], lvlmin=kw['lvlmin'],
                                      show_planet=kw["show_planet"], show_moon=kw["show_moon"],
                                      celest_colors=kw["celest_colors"])

                    vis.set_title(r"Particle Densities $log_{10} (N[\mathrm{cm}^{-2}])$ around Planetary Body", size=25)

                if scatter:
                    species = self.params.get_species(num=k + 1)
                    points = self._p_positions[np.where(self._p_species == species.id)]
                    dens, delaunay = self.dtfe(species, ts, d=2, grid=False, los=True)
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

    def plot3d(self, timestep, species_num=1, log_cutoff=None):
        self.__pull_data(timestep)

        pos = self._p_positions[self._sim_instance.N_active:]
        species = self.params.get_species(num=species_num)
        dens, _ = self.dtfe(species, timestep, d=3, grid=False)

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
        fig.update_layout(scene_aspectmode='cube')
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
            self.__pull_data(ts)
            self._sim_instance = self.sa[ts]

            dens2d, _ = self.dtfe(species, ts, d=2, grid=False, los=True)
            dens3d, _ = self.dtfe(species, ts, d=3, grid=False)

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

            print(ts)
            print(len(logDens2dInvSort[logDens2dInvSort > 0]))

            dens_max.append(logDens3dInvSort[0])
            #dens_mean.append(np.mean(np.array_split(logDens3dInvSort, 2)[0]))
            dens_mean.append(np.mean(logDens3dInvSort))
            los_max.append(logDens2dInvSort[0])
            #los_mean.append(np.mean(np.array_split(logDens2dInvSort, 2)[0]))
            los_mean.append(np.mean(logDens2dInvSort))

        return dens_max, dens_mean, los_max, los_mean

    def phase_curve(self, timesteps='auto', title='unnamed', fig=True, savefig=False, save_data=False,
                    load_path=None, column_dens=True, part_dens=True):
        import matplotlib.pyplot as plt
        # REBX: instances, rebx = self.sa
        first_instance = self.sa[0]
        advances_per_orbit = 1/self.params.int_spec["sim_advance"]

        if len(self.sa) < advances_per_orbit:
            print("No orbit has been completed.")
            return

        ts_list = []
        if timesteps == 'auto':
            second_instance_index = int(list(self.hash_supdict.keys())[1])
            second_instance = self.sa[second_instance_index]
            orbit_phase = np.around(second_instance.particles["moon"].calculate_orbit(
                primary=second_instance.particles["planet"]).theta * 180 / np.pi)
            orbit_first_index = len(self.sa) - (second_instance_index + 360 / orbit_phase * second_instance_index)

            if (orbit_first_index - orbit_first_index % 5 + 1 - 15) > second_instance_index:
                ts_list = np.arange(orbit_first_index - orbit_first_index % 5 + 1 - 15,
                                    len(self.sa) - len(self.sa) % 5 + 1, 5)
            else:
                ts_list = np.arange(orbit_first_index - orbit_first_index % 5 + 1,
                                    len(self.sa) - len(self.sa) % 5 + 1, 5)

        elif isinstance(timesteps, int):
            ts_list.append(timesteps)
        elif isinstance(timesteps, list):
            ts_list = timesteps
        elif isinstance(timesteps, np.ndarray):
            ts_list = np.ndarray.tolist(timesteps)
        else:
            raise TypeError("top-down timestep has an invalid type. Use 'int', 'list' or 'ndarray'")

        phases = []
        for timestep in ts_list:
            self.__pull_data(int(timestep))
            phase = self._sim_instance.particles["moon"].calculate_orbit(primary=self._sim_instance.particles["planet"]).theta * 180 / np.pi
            phases.append(phase)

        sort_index = np.argsort(phases)
        phases = np.asarray(phases)[sort_index]
        ts_list = ts_list[sort_index]

        shadow_phase = np.arctan2(first_instance.particles["planet"].r, first_instance.particles["moon"].calculate_orbit(primary=first_instance.particles["planet"]).a)
        los_ingress_phase = (np.pi - shadow_phase) * 180 / np.pi
        los_egress_phase = (np.pi + shadow_phase) * 180 / np.pi
        shadow_ingress_phase = (2 * np.pi - shadow_phase) * 180 / np.pi
        shadow_egress_phase = shadow_phase * 180 / np.pi

        if load_path is None:
            dens_max, dens_mean, los_max, los_mean = self.logDensities(ts_list)
        else:
            df = pd.read_pickle(f"./schedule_archive/phaseCurves/data/PhaseCurveData-{load_path[11:]}.pkl")
            phases = df["phases"].values
            los_mean = df["mean_los"].values
            dens_mean = df["mean_dens"].values

        if (fig or savefig) and (column_dens or part_dens):

            color1 = matplotlib.colors.to_hex('ivory')
            color2 = matplotlib.colors.to_hex('darkorange')

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

            rows = 0
            if column_dens:
                rows += 1
            if part_dens:
                rows += 1

            fig = plt.figure(figsize=(15, 5*rows))
            axs = []
            axs_index = 0
            for i in range(rows):
                ax = fig.add_subplot(rows, 1, i+1)
                axs.append(ax)

            #fig.suptitle(f"Phase-Density Curves: {title}", fontsize=20)

            if part_dens:
                axs[axs_index].plot(phases, dens_mean, color='red')
                axs[axs_index].set_ylabel(r"log $\bar{n}$ [cm$^{-3}$]", fontsize=18)
                axs[axs_index].set_xlim(left=0, right=360)
                axs[axs_index].tick_params(axis='both', which='major', labelsize=16)
                if rows == 1:
                    axs[axs_index].set_xlabel(r"exomoon phase $\phi \in$ [0, 360]$^\circ$", fontsize=18)
                axs_index += 1

            if column_dens:
                axs[axs_index].plot(phases, los_mean, color='red')
                axs[axs_index].set_ylabel(r"log $\bar{N}$ [cm$^{-2}$]", fontsize=18)
                axs[axs_index].set_xlim(left=0, right=360)
                axs[axs_index].set_xlabel(r"exomoon phase $\phi \in$ [0, 360]$^\circ$", fontsize=18)
                axs[axs_index].tick_params(axis='both', which='major', labelsize=16)

            if part_dens:
                axs[0].set_title(f"Phase-Density Curves: $\mathbf{{{title}}}$", fontsize=22, y=1.0, pad=-27)
            else:
                axs[0].set_title(f"Phase-Density Curves: $\mathbf{{{title}}}$", fontsize=22)

            colors1 = get_color_gradient(color1, color2, 20)
            colors2 = get_color_gradient(color2, color1, 20)

            for i in range(0, 19):
                first_range = np.linspace(shadow_egress_phase, los_ingress_phase, 20)
                second_range = np.linspace(los_egress_phase, shadow_ingress_phase, 20)
                for k, ax in enumerate(axs):
                    ax.axvspan(first_range[i], first_range[i + 1], facecolor=colors2[i], alpha=0.8)
                    ax.axvspan(second_range[i], second_range[i + 1], facecolor=colors1[i], alpha=0.8)
                    ax.axvspan(shadow_ingress_phase, 360, facecolor='dimgray', alpha=0.5, label="_"*i+"Shadow")
                    ax.axvspan(0, shadow_egress_phase, facecolor='dimgray', alpha=0.5)
                    if column_dens and k == axs_index:
                        axs[k].axvspan(los_ingress_phase, los_egress_phase, facecolor='black', alpha=0.5,
                                       hatch='/', fill=False, label="_" * i + "Hidden")
                    leg = ax.legend(loc='upper right', framealpha=1, fontsize=16)
                    for lh in leg.legendHandles:
                        lh.set_alpha(1)

            plt.tight_layout()

            if not os.path.exists(f'output/phaseCurves'):
                os.makedirs(f'output/phaseCurves')

            if savefig:
                plt.savefig(f'output/phaseCurves/{title}.png', bbox_inches='tight')
            else:
                plt.show()

        if save_data:
            d = {"phases": phases, "timestep": ts_list, "max_dens": dens_max, "max_los": los_max,
                 "mean_dens": dens_mean, "mean_los": los_mean}
            df = pd.DataFrame(data=d)

            if not os.path.exists(f'output/phaseCurves/data'):
                os.makedirs(f'output/phaseCurves/data')

            df.to_pickle(f"./output/phaseCurves/data/PhaseCurveData-{title}.pkl")













