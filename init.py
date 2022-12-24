from species import Species
from objects import *


class Parameters:
    _instance = None

    # Integration specifics
    int_spec = {
        "moon": True,
        "sim_advance": 1 / 40,
        "num_sim_advances": 280,
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

    celest = celestial_objects(int_spec["moon"], set=6)
    species = {}
    num_species = 0

    def __new__(cls):

        if cls._instance is None:
            # self.species[f"species1"] = Species("H", description="H--5.845kg/s--2500m/s", n_th=0, n_sp=500, mass_per_sec=5.845, model_smyth_v_b=2500, model_smyth_v_M=10000)  # 585, lifetime=2.26*86400
            # self.species[f"species1"] = Species("O2", description="O2--14.35kg/s--4700m/s", n_th=0, n_sp=500, mass_per_sec=14.35, model_smyth_v_b=4700, model_smyth_v_M=10000)  # 1435, lifetime=3.3*86400
            # cls.species[f"species1"] = Species("H2", description="H2--6.69kg/s--1200m/s", n_th=0, n_sp=1000, mass_per_sec=6.69, model_smyth_v_b=1200, model_smyth_v_M=10000)  # 669, lifetime=7*86400
            cls.species["species1"] = Species('Na', description='Na--2.6e12kg/s--2.5-30km/s--tau6.7min', n_th=0, n_sp=1000, mass_per_sec=2.6e12, lifetime=6.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)

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

    def __call__(self):
        return

    def get_species(self, name=None, id=None, num=None):

        if num is not None:
            return self.species[f"species{num}"]

        elif id is not None:
            for i in range(self.num_species):
                if self.species[f"species{i+1}"].id == id:
                    return self.species[f"species{i+1}"]
            return None

        elif name is not None:
            for i in range(self.num_species):

                if self.species[f"species{i+1}"].name == name:
                    return self.species[f"species{i+1}"]
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
        Parameters()


class NewParams:
    # TODO: As subclass
    """
    This class allows for the change of simulation parameters.

    Possible changes
    ----------------
    species : list
        New species to analyze.
        Signature:  Species(name: str, n_th: int, n_sp: int, mass_per_sec: float,
                            duplicate: int, beta: float, lifetime: float, n_e: float, sput_spec: dict)
            If attribute not passed to species, it is either set to 'None' or to default value.

    objects: dict
        Manipulate celestial objects.
        If passed 'objects: dict', a new source can be defined and object properties can be changed
        (see rebound.Particle properties)
            Example: {'Io': {'source': True, 'm': 1e23, 'a': 3e9}, 'Ganymede': {...}}

    moon: bool
        Change between moon and planet systems (e.g. Jovian to 55 Cnc)
        NOTE: It is currently only possible to change the chem. network via lifetime between systems.

    int_spec: dict
        update integration parameters. Only the specific dict parameter can be passed.

    therm_spec: dict
        update thermal evaporation parameters. Only the specific dict parameter can be passed.

    celest_set: int
        Change celestial object set according to objects.py.
        NOTE: Only works if 'moon: bool' is also defined.

    """

    def __init__(self, species=None, objects=None, moon=None, int_spec=None, therm_spec=None, celest_set=1):
        self.species = species
        self.object_instructions = objects
        self.moon = moon
        self.celest_set = celest_set
        self.int_spec = int_spec
        self.therm_spec = therm_spec

        Parameters.reset()

        if celest_set != 1 and moon is None:
            print("Please state if there is a moon for a new celestial body set using bool: 'moon'.")

    def __call__(self):
        Parameters()
        if self.species is not None:
            Parameters.modify_species(*self.species)

        if self.moon is not None:
            Parameters.modify_objects(moon=self.moon, set=self.celest_set)

        if isinstance(self.object_instructions, dict):
            for k1, v1 in self.object_instructions.items():
                if isinstance(v1, dict) and k1 in Parameters.celest:
                    source = v1.pop('source', False)
                    # False if v1 now empty:
                    if v1:
                        Parameters.modify_objects(object=k1, as_new_source=source, new_properties=v1)
                    else:
                        Parameters.modify_objects(object=k1, as_new_source=source)

        if self.therm_spec is not None or self.int_spec is not None:
            Parameters.modify_spec(self.int_spec, self.therm_spec)

        return

