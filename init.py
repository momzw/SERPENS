import pickle
from network import Network
from objects import *

class SpeciesSpecifics:

    def __init__(self, mass_number, id, type="neutral"):
        amu = 1.660539066e-27
        self.type = type
        self.mass_num = mass_number
        self.m = mass_number * amu
        self.id = id
        self.network = Network(self.id).network


class Species(SpeciesSpecifics):

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

        # Handle key word arguments:
        self.beta = kwargs.get("beta", 0)

        tau = kwargs.get("lifetime", None)
        if tau is not None:
            self.network = tau

        electron_density = kwargs.get("n_e", None)
        if electron_density is not None:
            self.network = Network(self.id, e_scaling=electron_density).network

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


class Parameters:
    _instance = None

    # Integration specifics
    int_spec = {
        "moon": False,
        "sim_advance": 1 / 40,
        "num_sim_advances": 5,
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

    celest = celestial_objects(int_spec["moon"])
    species = {}
    num_species = 0

    def __new__(cls):

        if cls._instance is None:
            # self.species[f"species1"] = Species("H", description="H--5.845kg/s--2500m/s", n_th=0, n_sp=500, mass_per_sec=5.845, model_smyth_v_b=2500, model_smyth_v_M=10000)  # 585, lifetime=2.26*86400
            # self.species[f"species1"] = Species("O2", description="O2--14.35kg/s--4700m/s", n_th=0, n_sp=500, mass_per_sec=14.35, model_smyth_v_b=4700, model_smyth_v_M=10000)  # 1435, lifetime=3.3*86400
            cls.species[f"species1"] = Species("H2", description="H2--6.69kg/s--1200m/s", n_th=0, n_sp=1000, mass_per_sec=6.69, model_smyth_v_b=1200, model_smyth_v_M=10000)  # 669, lifetime=7*86400
            cls.num_species = len(cls.species)

            cls._instance = object.__new__(cls)

        return cls._instance

    def __str__(self):
        s = "Integration specifics: \n" + f"\t {self.int_spec} \n"
        s += "Species: \n"
        for i in range(1, self.num_species + 1):
            s += f"\t {i}) " + str(vars(self.species[f"species{i}"])["description"]) + '\n'
            for k in vars(self.species[f"species{i}"]).keys():
                s += "\t \t" + f"{k} : {vars(self.species[f'species{i}'])[k]} \n"
        s += f"Thermal evaporation parameters: \n \t {self.therm_spec} \n"
        return s


    # TODO: Deprecate - no longer needed with species dict
    def get_species(self, name=None, id=None, num=None):

        if num is not None:
            return self.species[f"species{num}"]

        elif id is not None:
            for i in range(self.num_species):
                if self.species[f"species{1}"].id == id:
                    return self.species[f"species{1}"]
            return None

        elif name is not None:
            for i in range(self.num_species):
                if self.species[f"species{1}"].name == name:
                    return self.species[f"species{1}"]
            return None

        else:
            return


    @classmethod
    def modify_species(cls, *args):
        # Overwrite species if args supply species
        if len(args) != 0:
            cls.species = {}
            for index, arg in enumerate(args):
                cls.species[f"species{index + 1}"] = arg
            cls.num_species = len(args)
            print("Globally modified species.")

    @classmethod
    def modify_objects(cls, moon="default", set=1, object=None, as_new_source=False, new_properties=None):

        if isinstance(moon, bool):
            cls.int_spec["moon"] = moon
            cls.celest = celestial_objects(moon, set=set)

        if as_new_source and object is not None:
            temp = cls.celest["moon"]
            cls.celest["moon"] = cls.celest[object]
            cls.celest[object] = temp

            cls.celest["moon"]["hash"] = "moon"
            del cls.celest[object]["hash"]
            print("Globally modified source object.")

        if object is not None and type(new_properties) == dict:
            cls.celest[object].update(new_properties)

    @classmethod
    def modify_spec(cls, int_spec=None, therm_spec=None):
        if int_spec is not None:
            cls.int_spec.update(int_spec)
        if therm_spec is not None:
            cls.therm_spec.update(therm_spec)

    @staticmethod
    def reset(species=True, objects=True):
        if species:
            Parameters._instance = None
            Parameters.species = {}
        if objects:
            Parameters.celest = celestial_objects(Parameters.int_spec["moon"], set=1)
