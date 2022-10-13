import rebound
import numpy as np

class Network:
    def __init__(self, id):
        if id == 1:
            self.network = 6 * 60 * 60

        elif id == 2:
            self.network = 20 * 60 * 60

        elif id == 3:
            tau1 = 1/1.4e-7
            reagent1 = "H+"
            products1 = "O+ H"

            tau2 = 1/3.5e-9
            reagent2 = "S+"
            products2 = "O+ S"

            tau3 = 1/1.0e-7
            reagent3 = "S++"
            products3 = "O+ S+"

            tau4 = 1/5.1e-6
            reagent4 = "e"
            products4 = "O+ 2e"

            tau5 = 1/1.7e-8
            reagent5 = "y"
            products5 = "O+ e"

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5])
            products = np.array([products1, products2, products3, products4, products5])

            self.network = np.vstack((lifetimes, reagents, products)).T

        elif id == 4:
            tau1 = 1/9.2e-10
            reagent1 = "H+"
            products1 = "O2+ H"

            tau2 = 1/9.2e-10
            reagent2 = "H+"
            products2 = "O+ O H"

            tau3 = 1/7.4e-9
            reagent3 = "H+"
            products3 = "O+ O+ H+ 2e"

            tau4 = 1/9.2e-10
            reagent4 = "H2+"
            products4 = "O2+ H2"

            tau5 = 1/1.7e-7
            reagent5 = "O+"
            products5 = "O2+ O"

            tau6 = 1/1.9e-7
            reagent6 = "O+"
            products6 = "O O+ O"

            tau7 = 1/7.7e-8
            reagent7 = "O+"
            products7 = "O O+ O+ e"

            tau8 = 1/8.0e-9
            reagent8 = "O+"
            products8 = "O O++ O+ 2e"

            tau9 = 1/8.2e-8
            reagent9 = "O+"
            products9 = "O+ O+ O+ 2e"

            tau10 = 1/3.9e-8
            reagent10 = "O+"
            products10 = "O+ O++ O 2e"

            tau11 = 1/1.2e-7
            reagent11 = "S++"
            products11 = "O2+ S+"

            tau12 = 1/3.5e-6
            reagent12 = "e"
            products12 = "O O e"

            tau13 = 1/5.4e-6
            reagent13 = "e"
            products13 = "O2+ 2e"

            tau14 = 1/2.0e-6
            reagent14 = "e"
            products14 = "O+ O 2e"

            tau15 = 1/6.9e-9
            reagent15 = "e"
            products15 = "O++ O 3e"

            tau16 = 1/2.0e-7
            reagent16 = "y"
            products16 = "O O"

            tau17 = 1/3.0e-8
            reagent17 = "y"
            products17 = "O2+ e"

            tau18 = 1/8.5e-8
            reagent18 = "y"
            products18 = "O O+ e"

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9, tau10, tau11, tau12, tau13, tau14, tau15, tau16, tau17, tau18])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8, reagent9, reagent10, reagent11, reagent12, reagent13, reagent14, reagent15, reagent16, reagent17, reagent18])
            products = np.array([products1, products2, products3, products4, products5, products6, products7, products8, products9, products10, products11, products12, products13, products14, products15,products16, products17, products18])

            self.network = np.vstack((lifetimes, reagents, products)).T

        elif id == 5:
            tau1 = 1 / 5.4e-7
            reagent1 = "O+"
            products1 = "H+ O"

            tau2 = 1 / 3.0e-6
            reagent2 = "e"
            products2 = "H+ 2e"

            tau3 = 1 / 4.5e-9
            reagent3 = "y"
            products3 = "H+ e"

            lifetimes = np.array([tau1, tau2, tau3])
            reagents = np.array([reagent1, reagent2, reagent3])
            products = np.array([products1, products2, products3])

            self.network = np.vstack((lifetimes, reagents, products)).T

        elif id == 6:
            tau1 = 1 / 1.4e-7
            reagent1 = "H+"
            products1 = "H H2+"

            tau2 = 1 / 2.7e-7
            reagent2 = "O+"
            products2 = "O H2+"

            tau3 = 1 / 1.1e-7
            reagent3 = "S++"
            products3 = "S+ H2+"

            tau4 = 1 / 4.1e-6
            reagent4 = "e"
            products4 = "2e H2+"

            tau5 = 1 / 2.1e-7
            reagent5 = "e"
            products5 = "H+ H 2e"

            tau6 = 1 / 1.6e-6
            reagent6 = "e"
            products6 = "H H e"

            tau7 = 1 / 5.1e-9
            reagent7 = "y"
            products7 = "H H"

            tau8 = 1 / 3.1e-9
            reagent8 = "y"
            products8 = "H2+ e"

            tau9 = 1 / 6.9e-10
            reagent9 = "y"
            products9 = "H H+ e"

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8, reagent9])
            products = np.array([products1, products2, products3, products4, products5, products6, products7, products8, products9])

            self.network = np.vstack((lifetimes, reagents, products)).T

        elif id == 7:
            self.network = 1200 * 60 * 60

        elif id == 8:
            tau1 = 494 * 60 * 60
            reagent1 = "O+"
            products1 = "SO2+ O"

            tau2 = 455473 * 60 * 60
            reagent2 = "S+"
            products2 = "SO2+ S"

            tau3 = 390 * 60 * 60
            reagent3 = "S++"
            products3 = "SO2+ S+"

            tau4 = 61 * 3600
            reagent4 = "e"
            products4 = "SO2+ 2e"

            tau5 = 137 * 3600
            reagent5 = "e"
            products5 = "SO+ O 2e"

            tau6 = 8053 * 3600
            reagent6 = "e"
            products6 = "O+ SO 2e"

            tau7 = 1123 * 3600
            reagent7 = "e"
            products7 = "S+ O2 2e"

            tau8 = 1393 * 3600
            reagent8 = "e"
            products8 = "O2+ S 2e"

            tau9 = 1052300 * 3600
            reagent9 = "e"
            products9 = "O SO++ 3e"

            tau10 = 4093 * 3600
            reagent10 = "y"
            products10 = "SO2+ e"

            tau11 = 45 * 3600
            reagent11 = "y"
            products11 = "SO O"

            tau12 = 131 * 3600
            reagent12 = "y"
            products12 = "S O2"

            lifetimes = np.array(
                [tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9, tau10, tau11, tau12])
            reagents = np.array(
                [reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8, reagent9, reagent10,
                 reagent11, reagent12])
            products = np.array(
                [products1, products2, products3, products4, products5, products6, products7, products8, products9,
                 products10, products11, products12])

            self.network = np.vstack((lifetimes, reagents, products)).T


class SpeciesSpecifics:

    def __init__(self, mass_number, id, type="neutral"):
        amu = 1.660539066e-27
        self.type = type
        self.mass_num = mass_number
        self.m = mass_number * amu
        self.id = id

    def network(self):
        return Network(self.id).network


class Species(SpeciesSpecifics):

    # How to implement new species:
    # * Add network/lifetime in <class: Network>
    # * Add to implementedSpecies in <class: Species>

    def __init__(self, name = None, n_th = 0, n_sp = 0, mass_per_sec = None, **kwargs):
        self.implementedSpecies = {
            "Na": 1,
            "O": 3,
            "O2": 4,
            "S": 2,
            "H": 5,
            "H2": 6,
            "NaCl": 7,
            "SO2": 8
        }
        if name in self.implementedSpecies:
            if self.implementedSpecies[name] == 1:
                self.id = self.implementedSpecies[name]
                super().__init__(23, self.id)

            elif self.implementedSpecies[name] == 2:
                self.id = self.implementedSpecies[name]
                super().__init__(32, self.id)

            elif self.implementedSpecies[name] == 3:
                self.id = self.implementedSpecies[name]
                super().__init__(16, self.id)

            elif self.implementedSpecies[name] == 4:
                self.id = self.implementedSpecies[name]
                super().__init__(32, self.id)

            elif self.implementedSpecies[name] == 5:
                self.id = self.implementedSpecies[name]
                super().__init__(1, self.id)

            elif self.implementedSpecies[name] == 6:
                self.id = self.implementedSpecies[name]
                super().__init__(2, self.id)

            elif self.implementedSpecies[name] == 7:
                self.id = self.implementedSpecies[name]
                super().__init__(58.44, self.id)

            elif self.implementedSpecies[name] == 8:
                self.id = self.implementedSpecies[name]
                super().__init__(64, self.id)

        else:
            print(f"The species '{name}' has not been implemented.")
            return

        self.n_th = n_th
        self.n_sp = n_sp
        self.mass_per_sec = mass_per_sec
        self.name = name

        self.sput_spec= {
            "sput_model": kwargs.get("sput_model", 'smyth'),

            "model_maxwell_max": kwargs.get("model_maxwell_max", 3000),
            "model_wurz_inc_part_speed": kwargs.get("model_wurz_inc_part_speed", 5000),
            "model_wurz_binding_en": kwargs.get("model_wurz_binding_en", 2.89 * 1.602e-19),
            "model_wurz_inc_mass_in_amu": kwargs.get("model_wurz_inc_mass_in_amu", 23),
            "model_wurz_ejected_mass_in_amu": kwargs.get("model_wurz_ejected_mass_in_amu", 23),

            "model_smyth_v_b": kwargs.get("model_smyth_v_b", 4000),
            "model_smyth_v_M": kwargs.get("model_smyth_v_M", 60000),
            "model_smyth_a": kwargs.get("model_smyth_a", 7 / 3)
        }

    def particles_per_superparticle(self, mass):
        num = mass / self.m
        if not (self.n_th == 0 and self.n_sp == 0):
            num_per_sup = num / (self.n_th + self.n_sp)
        else:
            num_per_sup = num
        return num_per_sup

    def method_chain(self):
        return self


class Parameters:

    # Integration specifics
    # NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
    int_spec = {
        "moon": True,
        "sim_advance": 1 / 24,              # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
        "num_sim_advances": 64,             # Number of times the simulation advances.
        "stop_at_steady_state": False,
        "gen_max": None,                    # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.
        "r_max": 4                          # Maximal radial distance in units of source's semi-major axis. Particles beyond get removed from simulation.
    }

    # Thermal evaporation parameters
    therm_spec = {
        "source_temp_max": 130,  # 125, #2703,
        "source_temp_min": 90,  # 50, #1609,
        "spherical_symm_ejection": True,
    }

    def __init__(self):
        self.species1 = Species("O", n_th=0, n_sp=2*585, mass_per_sec=5.845, model_smyth_v_b = 2500)
        self.species2 = Species("O2", n_th=0, n_sp=2*1435, mass_per_sec=14.35, model_smyth_v_b = 4700)
        self.species3 = Species("H2", n_th=0, n_sp=2*669, mass_per_sec=6.69, model_smyth_v_b = 1000)

        self.num_species = len(locals()['self'].__dict__)

        # Sputtering model and shape parameters
        self.sput_spec_default = {
            "sput_model": 'smyth',    # Valid inputs: maxwell, wurz, smyth.

            "model_maxwell_max": 3000,

            "model_wurz_inc_part_speed": 5000,
            "model_wurz_binding_en": 2.89 * 1.602e-19,  # See table 1, in: Kudriavtsev Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".
            "model_wurz_inc_mass_in_amu": 23,
            "model_wurz_ejected_mass_in_amu": 23,

            "model_smyth_v_b": 4000,
            "model_smyth_v_M": 60000,
            "model_smyth_a": 7/3      # Speed distribution shape parameter
        }

    def __str__(self):
        s = "Integration specifics: \n" + f"\t {self.int_spec} \n"
        s += "Species: \n"
        for i in range(1, self.num_species+1):
            species = locals()['self'].__dict__
            s += "\t" + str(vars(species[f"species{i}"])) + "\n"
        s += f"Thermal evaporation parameters: \n \t {self.therm_spec} \n"
        s += f"Sputtering model and shape parameters: \n \t {self.sput_spec_default} \n"
        return s

    def get_species(self, num):
        return locals()['self'].__dict__[f"species{num}"]

    def get_species_by_name(self, name):
        for i in range(self.num_species):
            if locals()['self'].__dict__[f"species{i+1}"].name == name:
                return locals()['self'].__dict__[f"species{i+1}"]

    def get_species_by_id(self, id):
        for i in range(self.num_species):
            if locals()['self'].__dict__[f"species{i + 1}"].id == id:
                return locals()['self'].__dict__[f"species{i+1}"]


    # Particle emission position
    # ---------------------
    # Longitude and latitude distributions may be changed inside the 'create_particle' function.


def add_major_objects(sim, hash, primary_hash = "planet"):
    sim.add(m=4.799e22, a=6.709e8, e=0.009, inc=0.0082, primary=sim.particles[primary_hash], hash=hash)
    sim.particles[hash].r = 1560800
    sim.N_active += 1


def init3(additional_majors = False, moon = True):
    """
    This function initializes the basic REBOUND simulation structure and adds the first 3 major objects.
    These three objects are the host star, planet and moon.
    :return: rebound simulation object
    """
    sim = rebound.Simulation()
    # sim.automateSimulationArchive("archive.bin", walltime=60)
    sim.integrator = "whfast" # Fast and unbiased symplectic Wisdom-Holman integrator. Suitability not yet assessed.
    sim.collision = "direct" # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    sim.collision_resolve = "merge"

    # SI units:
    # ---------
    sim.units = ('m', 's', 'kg')
    sim.G = 6.6743e-11

    # CGS units:
    # ----------
    # sim.G = 6.67430e-8

    sim.dt = 500

    # PRELIMINARY: moon defines which objects to use!
    # ----------------------------------------------
    if moon:
        # labels = ["Sun", "Jupiter", "Io"]
        # sim.add(labels)      # Note: Takes current position in the solar system. Therefore more useful to define objects manually in the following.
        sim.add(m=1.988e30, hash="sun")
        sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=sim.particles["sun"], hash="planet")  # Omega=1.753, omega=4.78

        sim.particles["sun"].r = 696340000
        sim.particles["planet"].r = 69911000
    else:
        # 55 Cancri e
        # -----------
        sim.add(m=1.799e30, hash="sun")
        sim.add(m=4.77179e25, a=2.244e9, e=0.05, inc=0.00288, primary=sim.particles["sun"], hash="planet")

        sim.particles["sun"].r = 6.56e8
        sim.particles["planet"].r = 1.196e7
    # ----------------------------------------------

    if moon:
        #sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles["planet"], hash="moon")
        #sim.particles["moon"].r = 1821600

        sim.add(m=4.799e22, a=6.709e8, e=0.009, inc=0.0082, primary=sim.particles["planet"], hash="moon")
        sim.particles["moon"].r = 1560800

        sim.N_active = 3
    else:
        sim.N_active = 2

    if additional_majors:
        add_major_objects(sim, hash="europa")

    sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    # => sim.ri_whfast.safe_mode = 0

    sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    print("======================================")
    print("Initialized new simulation instance.")
    print("======================================")

    return

