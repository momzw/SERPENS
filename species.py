from network import Network


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

