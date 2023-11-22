from src.network import Network


class SpeciesSpecifics:
    """
    Base properties of a species given its mass number and id.
    """
    def __init__(self, mass_number, id, type="neutral"):
        amu = 1.660539066e-27
        self.type = type
        self.mass_num = mass_number
        self.m = mass_number * amu
        self.id = id
        self.network = Network(self.id).network


class Species(SpeciesSpecifics):
    """
    Implemented species for SERPENS.
    Species info contains mass number and an id associated to the species.
    """
    species_info = {
        "Na": (23, 1),
        "S": (32, 2),
        "O": (16, 3),
        "O2": (32, 4),
        "H": (1, 5),
        "H2": (2, 6),
        "NaCl": (58.44, 7),
        "SO2": (64, 8),
        "O+": (16, 9),
        "CO2": (44, 10),
        "K": (39, 11)
    }

    default_sput_spec = {
        "sput_model": 'smyth',
        "model_maxwell_max": 3000,
        "model_wurz_inc_part_speed": 5000,
        "model_wurz_binding_en": 2.89 * 1.602e-19,
        "model_wurz_inc_mass_in_amu": 23,
        "model_wurz_ejected_mass_in_amu": 23,
        "model_smyth_v_b": 4000,
        "model_smyth_v_M": 60000,
        "model_smyth_a": 7 / 3
    }

    def __init__(self, name=None, n_th=0, n_sp=0, mass_per_sec=None, duplicate=None, **kwargs):
        self.implementedSpecies = self.species_info.keys()

        if name in self.implementedSpecies:
            mass_number, species_id = self.species_info[name]
            super().__init__(mass_number, species_id)
        else:
            raise ValueError(f"The species '{name}' has not been implemented.")

        if duplicate is not None:
            self.id = self.id * 10 + duplicate
        self.duplicate = duplicate

        self.n_th = n_th
        self.n_sp = n_sp
        self.mass_per_sec = mass_per_sec
        self.name = name

        # Handle keyword arguments
        self.beta = kwargs.get("beta", 0)
        self.tau = kwargs.get("lifetime", None)
        self.tau_shielded = kwargs.get("shielded_lifetime", None)
        self.electron_density = kwargs.get("n_e", None)

        if self.electron_density is not None:
            self.network = Network(self.species_id, e_scaling=self.electron_density).network

        if self.tau is not None:
            self.network = self.tau

        self.description = kwargs.get("description", self.name)

        # Merge the provided sput_spec with the default_sput_spec
        self.sput_spec = {**self.default_sput_spec, **kwargs.get("sput_spec", {})}
        self.validate_sput_spec()

    def __str__(self):
        return f"Species {self.name}: \n\tMdot [kg/s] = {self.mass_per_sec}, \n\tlifetime [s] / network = {self.network}," \
               f"\n\tNumber of thermal superparticles = {self.n_th}," \
               f"\n\tNumber of sputtered superparticles = {self.n_sp}"

    def validate_sput_spec(self):
        valid_models = ["smyth", "maxwell", "wurz"]

        sput_model = self.sput_spec["sput_model"]
        if sput_model not in valid_models:
            raise ValueError(f"Invalid sputtering model: '{sput_model}'")

        # Validate and set other sputtering specifications
        if sput_model == "smyth":
            v_b = self.sput_spec.get("model_smyth_v_b", 0)
            v_M = self.sput_spec.get("model_smyth_v_M", 0)
            a = self.sput_spec.get("model_smyth_a", 0)

            if not (isinstance(v_b, (int, float)) and v_b >= 0):
                raise ValueError("Invalid value for model_smyth_v_b")
            if not (isinstance(v_M, (int, float)) and v_M >= v_b):
                raise ValueError("Invalid value for model_smyth_v_M")
            if not (isinstance(a, (int, float)) and a > 0):
                raise ValueError("Invalid value for model_smyth_a")

        elif sput_model == "wurz":
            inc_part_speed = self.sput_spec.get("model_wurz_inc_part_speed", 0)
            binding_en = self.sput_spec.get("model_wurz_binding_en", 0)
            inc_mass_in_amu = self.sput_spec.get("model_wurz_inc_mass_in_amu", 0)
            ejected_mass_in_amu = self.sput_spec.get("model_wurz_ejected_mass_in_amu", 0)

            if not (isinstance(inc_part_speed, (int, float)) and inc_part_speed >= 0):
                raise ValueError("Invalid value for model_wurz_inc_part_speed")
            if not (isinstance(binding_en, (int, float)) and binding_en >= 0):
                raise ValueError("Invalid value for model_wurz_binding_en")
            if not (isinstance(inc_mass_in_amu, (int, float)) and inc_mass_in_amu > 0):
                raise ValueError("Invalid value for model_wurz_inc_mass_in_amu")
            if not (isinstance(ejected_mass_in_amu, (int, float)) and ejected_mass_in_amu > 0):
                raise ValueError("Invalid value for model_wurz_ejected_mass_in_amu")

        elif sput_model == "maxwell":
            # Validate Maxwell model specifications
            pass

        # ... Add validation for other models

    def particles_per_superparticle(self, mass):
        num = mass / self.m
        if not (self.n_th == 0 and self.n_sp == 0):
            num_per_sup = num / (self.n_th + self.n_sp)
        else:
            num_per_sup = num
        return num_per_sup

