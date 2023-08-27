from src.species import Species
import json


class DefaultFields:

    species = {}
    with open('resources/input_parameters.txt') as f:
        data = f.read().splitlines(True)
        int_spec = json.loads(data[["integration_specifics" in s for s in data].index(True)])["integration_specifics"]
        therm_spec = json.loads(data[["thermal_evap_parameters" in s for s in data].index(True)])["thermal_evap_parameters"]
        for k, v in json.loads(data[-1]).items():
            species[f"{k}"] = Species(**v)
        num_species = len(species)

    with open('resources/objects.txt') as f:
        data = f.read().splitlines(True)
        celest = json.loads(data[["Jupiter" in s for s in data].index(True)])

    @classmethod
    def change_defaults(cls, **kwargs):
        cls.int_spec = kwargs.get("int_spec", cls.int_spec)
        cls.therm_spec = kwargs.get("therm_spec", cls.therm_spec)
        cls.celest = kwargs.get("celest", cls.celest)
        cls.species = kwargs.get("species", cls.species)
        cls.num_species = len(cls.species)


class Parameters:
    _instance = None

    int_spec = {}
    therm_spec = {}
    celest = None
    species = {}
    num_species = 0

    def __new__(cls):

        if cls._instance is None:
            cls.int_spec = DefaultFields.int_spec
            cls.therm_spec = DefaultFields.therm_spec
            cls.species = DefaultFields.species
            cls.celest = DefaultFields.celest
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
        s += f"Celestial objects: \n \t {self.celest} \n"
        return s

    def __call__(self):
        return self

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
            print("Species loaded.")

    @classmethod
    def modify_objects(cls, moon=True, celestial_name="Jupiter", object=None, as_new_source=False, new_properties=None):

        if isinstance(moon, bool):
            cls.int_spec["moon"] = moon
            with open('resources/objects.txt') as f:
                saved_objects = f.read().splitlines(True)
                cls.celest = json.loads(saved_objects[[f"{celestial_name}" in s for s in saved_objects].index(True)])

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
    def reset():
        Parameters._instance = None
        Parameters()


class NewParams:
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

    """

    def __init__(self, species=None, objects=None, moon=None, int_spec=None, therm_spec=None, celestial_name="Jupiter"):
        self.species = species
        self.object_instructions = objects
        self.moon = moon
        self.celestial_name = celestial_name
        self.int_spec = int_spec
        self.therm_spec = therm_spec

        Parameters.reset()

        if celestial_name != "Jupiter" and moon is None:
            print("Please state if there is a moon for a new celestial body set using bool: 'moon'.")

    def __call__(self):
        Parameters()
        if self.species is not None:
            Parameters.modify_species(*self.species)

        if self.moon is not None:
            Parameters.modify_objects(moon=self.moon, celestial_name=self.celestial_name)

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
            Parameters.modify_spec(int_spec=self.int_spec, therm_spec=self.therm_spec)

        return
