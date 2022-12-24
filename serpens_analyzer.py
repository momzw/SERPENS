import numpy as np
import os as os
import shutil
import rebound
import pickle
import dill
from datetime import datetime
import DTFE
import DTFE3D
from init import Parameters
from visualize import Visualize


class SerpensAnalyzer:

    def __init__(self, save_output=False, save_archive=False):
        # PICKLE:
        # ===================
        # print("WARNING: SERPENS is about to unpickle particle data. Pickle files are not secure. Make sure you trust the source!")
        # input("\t Press Enter to continue...")

        try:
            with open('hash_library.pickle', 'rb') as handle:
                self.hash_supdict = pickle.load(handle)

            with open('Parameters.pkl', 'rb') as handle:
                params_load = dill.load(handle)
                params_load()
        except:
            raise Exception("hash_library.pickle and/or Parameters.pkl not found.")

        try:
            self.sa = rebound.SimulationArchive("archive.bin", process_warnings=False)
        except:
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

        if save_output:
            print("Copying and saving...")
            if self.moon_exists:
                self.path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_moonsource"
            else:
                self.path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_planetsource"

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

    def __pull_data(self, timestep, return_grid = False):
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

            # self._p_weights[k1] = hash_dict_current[str(particle_hashes[k1])]["weight"]

        if return_grid:
            if self.moon_exists:
                boundary = self.params.int_spec["r_max"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                X, Y = np.meshgrid(np.linspace(-boundary + sim_instance.particles["planet"].x, boundary + sim_instance.particles["planet"].x, 100),
                                   np.linspace(-boundary + sim_instance.particles["planet"].y,
                                               boundary + sim_instance.particles["planet"].y, 100))
            else:
                boundary = self.params.int_spec["r_max"] * sim_instance.particles["planet"].a
                X, Y = np.meshgrid(np.linspace(-boundary, boundary, 100),
                                   np.linspace(-boundary, boundary, 100))
            return X, Y

    def dtfe(self, species, timestep, d=2, grid=True):

        X, Y = self.__pull_data(timestep, return_grid=True)
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

        total_injected = timestep * (species.n_sp + species.n_th)
        remaining_part = len(points[:,0])
        mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
        superpart_mass = mass_in_system / remaining_part
        number_per_superpart = superpart_mass / species.m
        phys_weights = number_per_superpart * weights

        print("Constructing DTFE ...")

        if d == 2:
            dtfe = DTFE.DTFE(points[:, :2], velocities[:, :2], phys_weights)
            if grid:
                dens = dtfe.density(X.flat, Y.flat).reshape((100, 100)) / 1e4
            else:
                dens = dtfe.density(points[:, 0], points[:, 1]) / 1e4
        elif d == 3:
            dtfe = DTFE3D.DTFE(points, velocities, superpart_mass)
            dens = dtfe.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6 * phys_weights
            if grid:
                print("Grid currently not available in 3d.")
        else:
            raise ValueError("Invalid dimension in DTFE.")

        print("\t ... done!")

        return dens, dtfe.delaunay

    def get_points(self, timestep):
        self.__pull_data(timestep)
        return self._p_positions

    def birdseye(self, timestep, d=3, grid=True, colormesh=True, scatter=False, triplot=True, show=True):
        # TOP DOWN COLUMN DENSITY PLOTS
        # ====================================
        self._sim_instance = self.sa[timestep]
        X, Y = self.__pull_data(timestep, return_grid=True)

        vis = Visualize(self._sim_instance, lim=10)

        for k in range(self.params.num_species):
            species = self.params.get_species(num=k + 1)
            points = self._p_positions[np.where(self._p_species == species.id)]
            dens, delaunay = self.dtfe(species, timestep, d=d, grid=grid)

            if colormesh:
                if grid and d==2:
                    vis.add_colormesh(k, X, Y, dens, contour=True, fill_contour=True, zorder=3, numlvls=25)
                else:
                    print("Please active the grid and set d = 2 for a colormesh.")
            if scatter:
                if not grid:
                    vis.add_densityscatter(k, points[:, 0], points[:, 1], dens, perspective="topdown", cb_format='%.2f', zorder=5, celest_colors=['y', 'sandybrown', 'b'])
                else:
                    print("Please deactivate the grid for a scatter plot.")
            if triplot:
                if d==3:
                    vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices[:,:3], perspective="topdown")
                elif d==2:
                    vis.add_triplot(k, points[:, 0], points[:, 1], delaunay.simplices, perspective="topdown", zorder=3)

            if d == 2:
                vis.set_title(r"Particle Densities $log_{10} (n[\mathrm{cm}^{-2}])$ around Planetary Body")
            elif d == 3:
                vis.set_title(r"Particle Densities $log_{10} (n[\mathrm{cm}^{-3}])$ around Planetary Body")

        if self.save:
            vis(show_bool=show, save_path=self.path, filename=f'{timestep}_{self.save_index}_td')
            self.save_index += 1
        else:
            vis(show_bool=show)
        del vis