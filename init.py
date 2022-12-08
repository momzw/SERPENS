import rebound
import reboundx
import numpy as np
import os

from network import Network


class Network:
    def __init__(self, id):
        self._species_weights = None
        u = 1.6726e-27
        e = 1.602e-19
        if id == 1:
            # Na
            self._network = 6 * 60 * 60

        elif id == 2:
            # S
            self._network = 20 * 60 * 60

        elif id == 3:
            # O

            # Charge-Exchange with protons at 50keV
            tau1 = 1 / 4.5e-6
            reagent1 = "H+"
            products1 = "O+ H"
            delv1 = np.sqrt(2 * 50e3 * e / u)

            # Charge-Exchange with protons at 60keV
            tau2 = 1 / 4.0e-6
            reagent2 = "H+"
            products2 = "O+ H"
            delv2 = np.sqrt(2 * 60e3 * e / u)

            # Charge-Exchange with protons at 70keV
            tau3 = 1 / 3.5e-7
            reagent3 = "H+"
            products3 = "O+ H"
            delv3 = np.sqrt(2 * 70e3 * e / u)

            # Charge-Exchange with protons at 80keV
            tau4 = 1 / 3.0e-7
            reagent4 = "H+"
            products4 = "O+ H"
            delv4 = np.sqrt(2 * 80e3 * e / u)

            tau5 = 1 / 3.5e-9
            reagent5 = "S+"
            products5 = "O+ S"
            delv5 = 0

            tau6 = 1 / 1.0e-7
            reagent6 = "S++"
            products6 = "O+ S+"
            delv6 = 0

            tau7 = 1 / 5.1e-6
            reagent7 = "e"
            products7 = "O+ 2e"
            delv7 = 0

            tau8 = 1 / 1.7e-8
            reagent8 = "y"
            products8 = "O+ e"
            delv8 = 0

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8])
            velocities = np.array([delv1, delv2, delv3, delv4, delv5, delv6, delv7, delv8])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8])
            products = np.array(
                [products1, products2, products3, products4, products5, products6, products7, products8])

            self._network = np.vstack((lifetimes, reagents, products, velocities)).T

        elif id == 4:
            # O2

            # Charge-Exchange with protons at 50keV
            t1 = 1 / 5.5e-6
            reag1 = "H+"
            prod1 = "O2+ H"
            dv1 = np.sqrt(2 * 50e3 * e / u)

            # Charge-Exchange with protons at 60keV
            t19 = 1 / 5.25e-6
            reag19 = "H+"
            prod19 = "O2+ H"
            dv19 = np.sqrt(2 * 60e3 * e / u)

            # Charge-Exchange with protons at 70keV
            t20 = 1 / 5.0e-6
            reag20 = "H+"
            prod20 = "O2+ H"
            dv20 = np.sqrt(2 * 70e3 * e / u)

            # Charge-Exchange with protons at 80keV
            t21 = 1 / 4.75e-6
            reag21 = "H+"
            prod21 = "O2+ H"
            dv21 = np.sqrt(2 * 80e3 * e / u)

            t2 = 1 / 9.2e-10
            reag2 = "H+"
            prod2 = "O+ O H"
            dv2 = 0

            t3 = 1 / 7.4e-9
            reag3 = "H+"
            prod3 = "O+ O+ H+ 2e"
            dv3 = 0

            t4 = 1 / 9.2e-10
            reag4 = "H2+"
            prod4 = "O2+ H2"
            dv4 = 0

            t5 = 1 / 1.7e-7
            reag5 = "O+"
            prod5 = "O2+ O"
            dv5 = 0

            t6 = 1 / 1.9e-7
            reag6 = "O+"
            prod6 = "O O+ O"
            dv6 = 0

            t7 = 1 / 7.7e-8
            reag7 = "O+"
            prod7 = "O O+ O+ e"
            dv7 = 0

            t8 = 1 / 8.0e-9
            reag8 = "O+"
            prod8 = "O O++ O+ 2e"
            dv8 = 0

            t9 = 1 / 8.2e-8
            reag9 = "O+"
            prod9 = "O+ O+ O+ 2e"
            dv9 = 0

            t10 = 1 / 3.9e-8
            reag10 = "O+"
            prod10 = "O+ O++ O 2e"
            dv10 = 0

            t11 = 1 / 1.2e-7
            reag11 = "S++"
            prod11 = "O2+ S+"
            dv11 = 0

            t12 = 1 / 3.5e-6
            reag12 = "e"
            prod12 = "O O e"
            dv12 = 0

            t13 = 1 / 5.4e-6
            reag13 = "e"
            prod13 = "O2+ 2e"
            dv13 = 0

            t14 = 1 / 2.0e-6
            reag14 = "e"
            prod14 = "O+ O 2e"
            dv14 = 0

            t15 = 1 / 6.9e-9
            reag15 = "e"
            prod15 = "O++ O 3e"
            dv15 = 0

            t16 = 1 / 2.0e-7
            reag16 = "y"
            prod16 = "O O"
            dv16 = 0

            t17 = 1 / 3.0e-8
            reag17 = "y"
            prod17 = "O2+ e"
            dv17 = 0

            t18 = 1 / 8.5e-8
            reag18 = "y"
            prod18 = "O O+ e"
            dv18 = 0

            lifetimes = np.array(
                [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21])
            reagents = np.array(
                [reag1, reag2, reag3, reag4, reag5, reag6, reag7, reag8, reag9, reag10, reag11, reag12, reag13, reag14,
                 reag15, reag16, reag17, reag18, reag19, reag20, reag21])
            products = np.array(
                [prod1, prod2, prod3, prod4, prod5, prod6, prod7, prod8, prod9, prod10, prod11, prod12, prod13, prod14,
                 prod15, prod16, prod17, prod18, prod19, prod20, prod21])
            delvs = np.array(
                [dv1, dv2, dv3, dv3, dv4, dv6, dv7, dv8, dv9, dv10, dv11, dv12, dv13, dv14, dv15, dv16, dv17, dv18,
                 dv19, dv20, dv21])

            self._network = np.vstack((lifetimes, reagents, products, delvs)).T

        elif id == 5:
            # H

            # Charge-Exchange with protons at 50keV
            tau4 = 1 / 2.0e-6
            reagent4 = "H+"
            products4 = "H+ H"
            delv4 = np.sqrt(2 * 50e3 * e / u)

            # Charge-Exchange with protons at 60keV
            tau5 = 1 / 9.975e-7
            reagent5 = "H+"
            products5 = "H+ H"
            delv5 = np.sqrt(2 * 60e3 * e / u)

            # Charge-Exchange with protons at 70keV
            tau6 = 1 / 9e-7
            reagent6 = "H+"
            products6 = "H+ H"
            delv6 = np.sqrt(2 * 70e3 * e / u)

            # Charge-Exchange with protons at 80keV
            tau7 = 1 / 6e-7
            reagent7 = "H+"
            products7 = "H+ H"
            delv7 = np.sqrt(2 * 80e3 * e / u)

            tau1 = 1 / 5.4e-7
            reagent1 = "O+"
            products1 = "H+ O"
            delv1 = 0

            tau2 = 1 / 3.0e-6
            reagent2 = "e"
            products2 = "H+ 2e"
            delv2 = 0

            tau3 = 1 / 4.5e-9
            reagent3 = "y"
            products3 = "H+ e"
            delv3 = 0

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7])
            products = np.array([products1, products2, products3, products4, products5, products6, products7])
            velocities = np.array([delv1, delv2, delv3, delv4, delv5, delv6, delv7])

            self._network = np.vstack((lifetimes, reagents, products, velocities)).T

        elif id == 6:
            # H2

            # Charge-Exchange with protons at 50keV
            tau1 = 1 / 4.0e-6
            reagent1 = "H+"
            products1 = "H H2+"
            delv1 = 0

            ## Charge-Exchange with protons at 60keV
            # tau10 = 1 / 3.0e-6
            # reagent10 = "H+"
            # products10 = "H H2+"
            # delv10 = np.sqrt(2 * 60e3 * e / u)
            #
            ## Charge-Exchange with protons at 70keV
            # tau11 = 1 / 2.0e-6
            # reagent11 = "H+"
            # products11 = "H H2+"
            # delv11 = np.sqrt(2 * 70e3 * e / u)
            #
            ## Charge-Exchange with protons at 80keV
            # tau12 = 1 / 1.0e-6
            # reagent12 = "H+"
            # products12 = "H H2+"
            # delv12 = np.sqrt(2 * 80e3 * e / u)

            tau2 = 1 / 2.7e-7
            reagent2 = "O+"
            products2 = "O H2+"
            delv2 = 0

            tau3 = 1 / 1.1e-7
            reagent3 = "S++"
            products3 = "S+ H2+"
            delv3 = 0

            tau4 = 1 / 4.1e-6
            reagent4 = "e"
            products4 = "2e H2+"
            delv4 = 0

            tau5 = 1 / 2.1e-7
            reagent5 = "e"
            products5 = "H+ H 2e"
            delv5 = 0

            tau6 = 1 / 1.6e-6
            reagent6 = "e"
            products6 = "H H e"
            delv6 = 0

            tau7 = 1 / 5.1e-9
            reagent7 = "y"
            products7 = "H H"
            delv7 = 0

            tau8 = 1 / 3.1e-9
            reagent8 = "y"
            products8 = "H2+ e"
            delv8 = 0

            tau9 = 1 / 6.9e-10
            reagent9 = "y"
            products9 = "H H+ e"
            delv9 = 0

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9])
            reagents = np.array(
                [reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8, reagent9])
            products = np.array(
                [products1, products2, products3, products4, products5, products6, products7, products8, products9])
            velocities = np.array(
                [delv1, delv2, delv3, delv4, delv5, delv6, delv7, delv8, delv9])

            self._network = np.vstack((lifetimes, reagents, products, velocities)).T

        elif id == 7:
            # NaCl
            self._network = 60 * 60 * 24 * 5

        elif id == 8:
            # SO2

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

            self._network = np.vstack((lifetimes, reagents, products)).T

        elif id == 9:
            # O+

            # Charge-Exchange with protons at 50keV
            t1 = 1 / 3.0e-7
            reag1 = "H+"
            prod1 = "H"
            dv1 = np.sqrt(2 * 50e3 * e / (16 * u))

            # Charge-Exchange with protons at 60keV
            t2 = 1 / 2.8e-7
            reag2 = "H+"
            prod2 = "H"
            dv2 = np.sqrt(2 * 60e3 * e / (16 * u))

            # Charge-Exchange with protons at 70keV
            t3 = 1 / 2.0e-7
            reag3 = "H+"
            prod3 = "H"
            dv3 = np.sqrt(2 * 70e3 * e / (16 * u))

            # Charge-Exchange with protons at 80keV
            t4 = 1 / 1.0e-7
            reag4 = "H+"
            prod4 = "H"
            dv4 = np.sqrt(2 * 80e3 * e / (16 * u))

            lifetimes = np.array([t1, t2, t3, t4])
            reagents = np.array([reag1, reag2, reag3, reag4])
            products = np.array([prod1, prod2, prod3, prod4])
            delvs = np.array([dv1, dv2, dv3, dv4])

            self._network = np.vstack((lifetimes, reagents, products, delvs)).T

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, tau):
        if isinstance(tau, (float, int)):
            self._network = tau
        else:
            print("Could not set network lifetime")


class SpeciesSpecifics:

    def __init__(self, mass_number, id, type="neutral"):
        amu = 1.660539066e-27
        self.type = type
        self.mass_num = mass_number
        self.m = mass_number * amu
        self.id = id
        self.network = Network(self.id).network


class Species(SpeciesSpecifics):

    # How to implement new species:
    # * Add network/lifetime in <class: Network>
    # * Add to implementedSpecies in <class: Species>

    def __init__(self, name=None, n_th=0, n_sp=0, mass_per_sec=None, duplicate=None, **kwargs):
        self.implementedSpecies = {
            "Na": 1,
            "O": 3,
            "O2": 4,
            "S": 2,
            "H": 5,
            "H2": 6,
            "NaCl": 7,
            "SO2": 8,
            "O+": 9
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

            elif self.implementedSpecies[name] == 9:
                self.id = self.implementedSpecies[name]
                super().__init__(16, self.id)

        else:
            print(f"The species '{name}' has not been implemented.")
            return

        if duplicate is not None:
            self.id = self.id * 10 + duplicate
        self.duplicate = duplicate

        self.n_th = n_th
        self.n_sp = n_sp
        self.mass_per_sec = mass_per_sec
        self.name = name

        self.beta = kwargs.get("beta", 0)

        tau = kwargs.get("lifetime", None)
        if tau is not None:
            self.network = tau

        self.description = kwargs.get("description", self.name)

        self.sput_spec = {
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
    # TODO: Think about positions <-> species (id) ?

    # Integration specifics
    # NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
    int_spec = {
        "moon": True,
        "sim_advance": 1 / 40,
        "num_sim_advances": 480,
        "stop_at_steady_state": False,
        "gen_max": None,
        "r_max": 4,
        "random_walk": False,
        "particle_interpolation": False
    }

    # Thermal evaporation parameters
    therm_spec = {
        "source_temp_max": 2703,  # 125, #2703, 130
        "source_temp_min": 1609,  # 50, #1609, 90
        "spherical_symm_ejection": True,
    }

    def __init__(self):
        #self.species1 = Species("H", description="H--5.845kg/s--2500m/s", n_th=0, n_sp=500, mass_per_sec=5.845, model_smyth_v_b=2500, model_smyth_v_M=10000)  # 585, lifetime=2.26*86400
        #self.species1 = Species("O2", description="O2--14.35kg/s--4700m/s", n_th=0, n_sp=500, mass_per_sec=14.35, model_smyth_v_b=4700, model_smyth_v_M=10000)  # 1435, lifetime=3.3*86400
        self.species1 = Species("H2", description="H2--6.69kg/s--1200m/s", n_th=0, n_sp=1000, mass_per_sec=6.69, model_smyth_v_b=1200, model_smyth_v_M=10000)  # 669, lifetime=7*86400

        # self.species1 = Species("SO2", description="SO2-30km/s", n_th=0, n_sp=1500, mass_per_sec=1000e3, model_smyth_v_b=30000, beta=.1, lifetime=17.7*360)
        # self.species2 = Species("NaCl", description="NaCl-30km/s-1hr", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=30000, duplicate=1, lifetime=60*60)
        # self.species3 = Species("NaCl", description="NaCl-30km/s-5d", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=30000, duplicate=2)
        # self.species4 = Species("NaCl", description="NaCl-10km/s-5d", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=10000, duplicate=3)
        # self.species5 = Species("NaCl", description="NaCl-50km/s-5d", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=50000, duplicate=4)
        # self.species6 = Species("NaCl", description="NaCl-30km/s-5d-RAD", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=30000, duplicate=5, beta=0.1)
        # self.species7 = Species("NaCl", description="NaCl-30km/s-5d-LRAD", n_th=0, n_sp=300, mass_per_sec=1000, model_smyth_v_b=30000, duplicate=6, beta=1)

        self.num_species = len(locals()['self'].__dict__)

        # Sputtering model and shape parameters
        self.sput_spec_default = {
            "sput_model": 'smyth',  # Valid inputs: maxwell, wurz, smyth.

            "model_maxwell_max": 3000,

            "model_wurz_inc_part_speed": 5000,
            "model_wurz_binding_en": 2.89 * 1.602e-19,
            # See table 1, in: Kudriavtsev Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".
            "model_wurz_inc_mass_in_amu": 23,
            "model_wurz_ejected_mass_in_amu": 23,

            "model_smyth_v_b": 4000,
            "model_smyth_v_M": 98000,
            "model_smyth_a": 7 / 3  # Speed distribution shape parameter
        }

    def __str__(self):
        s = "Integration specifics: \n" + f"\t {self.int_spec} \n"
        s += "Species: \n"
        for i in range(1, self.num_species + 1):
            species = locals()['self'].__dict__
            s += "\t" + str(vars(species[f"species{i}"])) + "\n"
        s += f"Thermal evaporation parameters: \n \t {self.therm_spec} \n"
        return s

    def get_species(self, name=None, id=None, num=None):
        if num != None:
            return locals()['self'].__dict__[f"species{num}"]
        elif id != None:
            for i in range(self.num_species):
                if locals()['self'].__dict__[f"species{i + 1}"].id == id:
                    return locals()['self'].__dict__[f"species{i + 1}"]
            return None
        elif name != None:
            for i in range(self.num_species):
                if locals()['self'].__dict__[f"species{i + 1}"].name == name:
                    return locals()['self'].__dict__[f"species{i + 1}"]
            return None
        else:
            return
