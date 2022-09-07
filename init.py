import rebound

class SpeciesSpecifics:

    def __init__(self, mass_number, type="neutral"):
        amu = 1.660539066e-27
        self.type = type
        self.mass_num = mass_number
        self.m = mass_number * amu


class Species(SpeciesSpecifics):

    def __init__(self, element, n_th = None, n_sp = None):
        self.element = element
        implementedSpecies = {
            "Sodium": 1,
            "Oxygen": 2,
            "Sulfur": 3
        }
        if element in implementedSpecies:
            if implementedSpecies[element] == 1:
                super().__init__(23)
                self.bind = 2.89 * 1.602e-19
                self.id = 1
                self.lifetime = 2 * 60 * 60
            elif implementedSpecies[element] == 2:
                super().__init__(16)
                self.id = 2
                self.lifetime = 20 * 60 * 60
            else:
                super().__init__(32)
                self.id = 3
                self.lifetime = 40 * 60 * 60
        else:
            print(f"The species '{element}' has not been implemented.")
            return

        self.n_th = n_th
        self.n_sp = n_sp

    def particles_per_superparticle(self, mass_per_sec):
        num = mass_per_sec / self.m
        return num / (self.n_th + self.n_sp)

    #def lifetime(self):
    #    tau = 17.7/2 * 60 * 60
    #    return tau

    def method_chain(self):
        return self


class Parameters:

    # Integration specifics
    # NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
    int_spec = {
        "moon": True,
        "sim_advance": 1 / 8,              # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
        "num_sim_advances": 64,             # Number of times the simulation advances.
        "stop_at_steady_state": False,
        "gen_max": None,                    # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.
        "r_max": 2                          # Maximal radial distance in units of source's semi-major axis. Particles beyond get removed from simulation.
    }

    def __init__(self):
        self.species1 = Species("Sodium", n_th=0, n_sp=500)
        self.species2 = Species("Oxygen", n_th=0, n_sp=500)
        #self.species3 = Species("Sulfur", n_th=0, n_sp=500)

        self.num_species = len(locals()['self'].__dict__)

        # Thermal evaporation parameters
        self.therm_spec = {
            "source_temp_max": 2703,
            "source_temp_min": 1609,
            "spherical_symm_ejection": False,
        }

        # Sputtering model and shape parameters
        self.sput_spec = {
            "sput_model": 'maxwell',    # Valid inputs: maxwell, wurz, smyth.

            "model_maxwell_mean": 2300,
            "model_maxwell_std": 200,

            "model_wurz_inc_part_speed": 5000,
            "model_wurz_binding_en": 2.89 * 1.602e-19,  # See table 1, in: Kudriavtsev Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".
            "model_wurz_inc_mass_in_amu": 23,
            "model_wurz_ejected_mass_in_amu": 23,

            "model_smyth_v_b": 1000,    # "low cutoff" speed to prevent the slowest nonescaping atoms from dominating the distribution (see Wilson et al. 2002)
            "model_smyth_v_M": 40000,   # Maximum velocity achievable. Proportional to plasma velocity (see Wilson et al. 2002)
            "model_smyth_a": 7 / 3      # Speed distribution shape parameter
        }

    def __str__(self):
        s = "Integration specifics: \n" + f"\t {self.int_spec} \n"
        s += "Species: \n"
        for i in range(1, self.num_species+1):
            species = locals()['self'].__dict__
            s += "\t" + str(vars(species[f"species{i}"])) + "\n"
        s += f"Thermal evaporation parameters: \n \t {self.therm_spec} \n"
        s += f"Sputtering model and shape parameters: \n \t {self.sput_spec} \n"
        return s

    def get_species(self, id):
        if id == 1:
            return self.species1.method_chain()
        elif id == 2:
            return self.species2.method_chain()
        elif id == 3:
            return self.species3.method_chain()
        else:
            return None

    # Particle emission position
    # ---------------------
    # Longitude and latitude distributions may be changed inside the 'create_particle' function.


def add_major_objects(sim, hash = None, primary_hash = "planet"):
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
        sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles["planet"], hash="moon")
        sim.particles["moon"].r = 1821600
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
    print("Initialized new simulation instance.")

    return

