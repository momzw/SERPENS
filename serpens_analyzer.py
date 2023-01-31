import numpy as np
import os as os
import shutil
import rebound
import dill
import matplotlib.cm as cm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src import DTFE, DTFE3D
from parameters import Parameters
from src.visualize import Visualize


class SerpensAnalyzer:

    def __init__(self, save_output=False, save_archive=False, folder_name=None, z_cutoff=None, r_cutoff=None):
        # PICKLE:
        # ===================
        # print("WARNING: SERPENS is about to unpickle particle data.
        # Pickle files are not secure. Make sure you trust the source!")
        # input("\t Press Enter to continue...")

        try:
            with open('hash_library.pickle', 'rb') as handle:
                self.hash_supdict = dill.load(handle)

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

        self.z_cutoff = z_cutoff
        self.r_cutoff = r_cutoff

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
        sim_instance = self.sa[timestep]
        if self.moon_exists:
            boundary = self.params.int_spec["r_max"] * sim_instance.particles["moon"].calculate_orbit(
                primary=sim_instance.particles["planet"]).a

            if plane == 'xy':
                offsetx = sim_instance.particles["planet"].x
                offsety = sim_instance.particles["planet"].y
                offsetz = 0
            elif plane == 'yz':
                offsetx = sim_instance.particles["planet"].y
                offsety = sim_instance.particles["planet"].z
                offsetz = 0
            elif plane == '3d':
                offsetx = sim_instance.particles["planet"].x
                offsety = sim_instance.particles["planet"].y
                offsetz = sim_instance.particles["planet"].z
            else:
                raise ValueError("Invalid plane in grid construction!")

        else:
            boundary = self.params.int_spec["r_max"] * sim_instance.particles["planet"].a
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
        sim_instance = self.sa[timestep]
        self._p_positions = np.zeros((sim_instance.N, 3), dtype="float64")
        self._p_velocities = np.zeros((sim_instance.N, 3), dtype="float64")
        self._p_hashes = np.zeros(sim_instance.N, dtype="uint32")
        self._p_species = np.zeros(sim_instance.N, dtype="int")
        self._p_weights = np.zeros(sim_instance.N, dtype="float64")
        sim_instance.serialize_particle_data(xyz=self._p_positions, vxvyvz=self._p_velocities,
                                             hash=self._p_hashes)
        if not timestep == 0:
            hash_dict_current = self.hash_supdict[str(timestep)]
        else:
            hash_dict_current = {}

        for k1 in range(sim_instance.N_active, sim_instance.N):
            self._p_species[k1] = hash_dict_current[str(self._p_hashes[k1])]["id"]
            particle_iter = hash_dict_current[str(self._p_hashes[k1])]["i"]

            if self.moon_exists:
                particle_time = (timestep - particle_iter) * self.params.int_spec["sim_advance"] * \
                                sim_instance.particles[
                                    "moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
            else:
                particle_time = (timestep - particle_iter) * self.params.int_spec["sim_advance"] * \
                                sim_instance.particles[
                                    "planet"].P

            if self.params.get_species(id=self._p_species[k1]) is None:
                continue

            chem_network = self.params.get_species(id=self._p_species[k1]).network
            reaction_rate = 0
            if not isinstance(chem_network, (int, float)):
                for l in range(np.size(chem_network[:, 0])):
                    reaction_rate += 1 / float(chem_network[:, 0][l])
            else:
                reaction_rate = 1 / chem_network
            self._p_weights[k1] = np.exp(-particle_time * reaction_rate)

            #self._p_weights[k1] = hash_dict_current[str(self._p_hashes[k1])]["weight"]

        if self.z_cutoff is not None:
            assert isinstance(self.z_cutoff, (float, int))
            mask = (self._p_positions[:,2] < self.z_cutoff * sim_instance.particles["planet"].r) \
                   & (self._p_positions[:,2] > -self.z_cutoff * sim_instance.particles["planet"].r)
            self._p_positions = self._p_positions[mask]
            self._p_velocities = self._p_velocities[mask]
            self._p_hashes = self._p_hashes[mask]
            self._p_species = self._p_species[mask]
            self._p_weights = self._p_weights[mask]

        if self.r_cutoff is not None:
            assert isinstance(self.z_cutoff, (float, int))
            r = np.linalg.norm(self._p_positions[:,:2] - sim_instance.particles["planet"].xyz[:2], axis=1)
            mask = r < self.r_cutoff * sim_instance.particles["planet"].r
            self._p_positions = self._p_positions[mask]
            self._p_velocities = self._p_velocities[mask]
            self._p_hashes = self._p_hashes[mask]
            self._p_species = self._p_species[mask]
            self._p_weights = self._p_weights[mask]

    def dtfe(self, species, timestep, d=2, grid=True, los=False):
        self.__pull_data(timestep)
        sim_instance = self.sa[timestep]

        if self.moon_exists:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
        else:
            simulation_time = timestep * self.params.int_spec["sim_advance"] * sim_instance.particles["planet"].P

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

        number_of_particles = 1e39

        phys_weights = number_of_particles * weights/np.sum(weights)

        if d == 2:

            if los:
                los_dist_to_planet = np.sqrt((points[:, 1] - self._sim_instance.particles["planet"].y) ** 2 +
                                             (points[:, 2] - self._sim_instance.particles["planet"].z) ** 2)
                mask = (los_dist_to_planet > self._sim_instance.particles["planet"].r) | (points[:, 0] - self._sim_instance.particles["planet"].x > 0)

                dtfe = DTFE.DTFE(points[:, 1:3][mask], velocities[:, 1:3][mask], phys_weights[mask])
                if grid:
                    Y, Z = self.__grid(timestep, plane='yz')
                    dens = dtfe.density(Y.flat, Z.flat).reshape((100, 100)) / 1e4
                else:
                    dens = dtfe.density(points[:, 1][mask], points[:, 2][mask]) / 1e4

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

        for ts in ts_list:
            self.__pull_data(ts)
            self._sim_instance = self.sa[ts]

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
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, perspective="topdown", cb_format='%.2f', zorder=1, celest_colors=kw["celest_colors"])

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
            else:
                vis(show_bool=show)

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

        for ts in ts_list:
            self.__pull_data(ts)
            self._sim_instance = self.sa[ts]

            vis = Visualize(self._sim_instance, lim=kw["lim"], cmap=kw["colormap"])

            for k in range(self.params.num_species):
                species = self.params.get_species(num=k + 1)

                if colormesh:
                    dens, delaunay = self.dtfe(species, ts, d=2, grid=True, los=True)

                    Y, Z = self.__grid(ts, plane='yz')
                    self.__pull_data(ts)
                    vis.add_colormesh(k, Y, Z, dens, contour=True, fill_contour=True, zorder=9, numlvls=25, perspective='los',
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
                    vis.add_densityscatter(k, points[:, 1][mask], points[:, 2][mask], dens, perspective="los", cb_format='%.2f',
                                           zorder=5, celest_colors=kw["celest_colors"],
                                           show_planet=kw["show_planet"], show_moon=kw["show_moon"])


            if self.save:
                vis(show_bool=show, save_path=self.path, filename=f'LOS_{ts}_{self.save_index}')
                self.save_index += 1
            else:
                vis(show_bool=show)

            del vis

    def plot3d(self, timestep, species_num=1, log_cutoff=5):
        self.__pull_data(timestep)
        sim_instance = self.sa[timestep]
        pos = self._p_positions[3:]
        species = self.params.get_species(num=species_num)
        dens, _ = self.dtfe(species, timestep, d=3, grid=False)

        phi, theta = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
        x = sim_instance.particles["planet"].r * np.sin(theta) * np.cos(phi)
        y = sim_instance.particles["planet"].r * np.sin(theta) * np.sin(phi)
        z = sim_instance.particles["planet"].r * np.cos(theta)

        np.seterr(divide='ignore')

        df = pd.DataFrame({
            'x': pos[:, 0][np.log10(dens) > log_cutoff],
            'y': pos[:, 1][np.log10(dens) > log_cutoff],
            'z': pos[:, 2][np.log10(dens) > log_cutoff]
        })

        fig = px.scatter_3d(df, x='x', y='y', z='z', color=np.log10(dens[np.log10(dens) > log_cutoff]), opacity=.5)
        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=x ** 2 + y ** 2 + z ** 2))
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
            ts_list = timestep
        elif isinstance(timestep, np.ndarray):
            ts_list = np.ndarray.tolist(timestep)
        else:
            raise TypeError("top-down timestep has an invalid type.")

        for ts in ts_list:
            self.__pull_data(ts)
            self._sim_instance = self.sa[ts]

            dens2d, _ = self.dtfe(species, ts, d=2, grid=True, los=True)
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

            dens_max.append(logDens3dInvSort[0])
            dens_mean.append(np.mean(logDens3dInvSort))
            los_max.append(logDens2dInvSort[0])
            los_mean.append(np.mean(logDens2dInvSort))

        return dens_max, dens_mean, los_max, los_mean

    def testground(self):
        advances_per_orbit = 1/self.params.int_spec["sim_advance"]
        if len(self.sa) < advances_per_orbit:
            print("No orbit has been completed.")
            return














