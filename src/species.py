#from src.network import Network


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
        # Deprecated.
        #self.network = Network(self.id).network


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
        "K": (39, 11),
        "Na+": (24, 1)
    }

    default_sput_spec = {
        "sput_model": 'smyth',
        "model_maxwell_max": 3000,
        "v_b": 4000,
        "v_M": 60000,
        "a": 7 / 3
    }

    def __init__(self, name=None, n_th=0, n_sp=0, mass_per_sec=None, duplicate=None, **kwargs):
        if name in self.species_info.keys():
            mass_number, species_id = self.species_info[name]
            super().__init__(mass_number, species_id)
        else:
            raise ValueError(f"The species '{name}' has not been implemented.")

        # `type_id` identifies the chemical species (used for reaction networks).
        # `id` is the per-variant identifier (used for particle tagging & analysis).
        self.type_id = self.id

        if duplicate is not None:
            # Keep the legacy scheme (base_id * 10 + duplicate) but only apply it to the variant id.
            self.id = self.type_id * 10 + duplicate
        self.duplicate = duplicate

        self.n_th = n_th
        self.n_sp = n_sp
        self.mass_per_sec = mass_per_sec
        self.name = name

        # Handle keyword arguments
        self.beta = kwargs.get("beta", 0)
        self.tau = kwargs.get("lifetime", 60 * 60) # 1 hour default
        self.tau_shielded = kwargs.get("shielded_lifetime", self.tau)
        self.electron_density = kwargs.get("n_e", None)

        # Particle charge [C] -> Lorentz force
        e = 1.602176634e-19
        charge_in_e = kwargs.get("charge_e", None)  # e.g. +1 for singly-ionized
        charge_c = kwargs.get("charge", None)  # directly in Coulombs
        if charge_c is not None:
            self.q = float(charge_c)
        elif charge_in_e is not None:
            self.q = float(charge_in_e) * e
        else:
            # sensible default: neutral unless the name hints at charge
            self.q = e if str(self.name).endswith("+") else 0.0

        # Recompute electron-dependent network using the chemical type id.
        # Deprecated: use `tau` instead.
        if self.electron_density is not None:
            #self.network = Network(self.type_id, e_scaling=self.electron_density).network
            pass

        # Deprecated: use `tau` instead.
        #if self.tau is not None:
        #    self.network = self.tau

        self.description = kwargs.get("description", self.name)

        # Merge the provided sput_spec with the default_sput_spec
        self.sput_spec = {**self.default_sput_spec, **kwargs.get("sput_spec", {})}
        self.validate_sput_spec()

        # Add reaction information
        self._reactions = kwargs.get("reactions", [])
        if isinstance(self._reactions, Reaction):
            self._reactions = [self._reactions]

    def __str__(self):
        return f"Species {self.name}: \n\tMdot [kg/s] = {self.mass_per_sec}, \n\tlifetime [s] = {self.tau}," \
               f"\n\tNumber of thermal superparticles = {self.n_th}," \
               f"\n\tNumber of sputtered superparticles = {self.n_sp}"

    def copy(self):
        new_species = Species()

    def validate_sput_spec(self):
        valid_models = ["smyth", "maxwell"]

        sput_model = self.sput_spec["sput_model"]
        if sput_model not in valid_models:
            raise ValueError(f"Invalid sputtering model: '{sput_model}'")

        # Validate and set other sputtering specifications
        if sput_model == "smyth":
            v_b = self.sput_spec.get("v_b", 0)
            v_M = self.sput_spec.get("v_M", 0)
            a = self.sput_spec.get("a", 0)

            if not (isinstance(v_b, (int, float)) and v_b >= 0):
                raise ValueError("Invalid value for v_b")
            if not (isinstance(v_M, (int, float)) and v_M >= v_b):
                raise ValueError("Invalid value for v_M")
            if not (isinstance(a, (int, float)) and a > 0):
                raise ValueError("Invalid value for a")

        elif sput_model == "maxwell":
            # Validate Maxwell model specifications
            print("Maxwell model not yet implemented.")
            pass

    def particles_per_superparticle(self, mass):
        num = mass / self.m
        if not (self.n_th == 0 and self.n_sp == 0):
            num_per_sup = num / (self.n_th + self.n_sp)
        else:
            num_per_sup = num
        return num_per_sup

    @property
    def reactions(self):
        return self._reactions

    @reactions.setter
    def reactions(self, reaction):
        if isinstance(reaction, Reaction):
            self._reactions.append(reaction)
        elif isinstance(reaction, list):
            assert all([isinstance(r, Reaction) for r in reaction])
            self._reactions.extend(reaction)
        else:
            print("Could not set/expand reaction network.")



class Reaction:
    def __init__(self, target_species_name, lifetime, **lifetime_kwargs):
        self.target_species_name = target_species_name
        self.lifetime = lifetime
        self.lifetime_kwargs = lifetime_kwargs

    def get_lifetime(self, pos):
        if callable(self.lifetime):
            return self.lifetime(*pos, **self.lifetime_kwargs)
        return self.lifetime

