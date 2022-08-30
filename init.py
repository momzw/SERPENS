import rebound


class Simulation_Parameters:

    def __init__(self):
        pass

    def int(self):
        # Integration specifics
        # NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
        self.int_spec = {
            "sim_advance": 1/12,            # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
            "num_sim_advances": 25,         # Number of times the simulation advances.
            "stop_at_steady_state": False,
            "gen_max": None                 # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.
        }
        return self.int_spec

    def gen(self):
        # Generating particles
        self.gen_spec = {
            "n_th": 0,      # Number of particles created by thermal evap each integration advance.
            "n_sp": 500,   # Number of particles created by sputtering each integration advance.
            "r_max": 2      # Maximal radial distance in units of moon's semi-major axis. Particles beyond get removed from simulation.
        }
        return self.gen_spec

    def therm(self):
        # Thermal evaporation parameters
        self.therm_spec = {
            "Io_temp_max": 130,
            "Io_temp_min": 90,
            "spherical_symm_ejection": False,
            "part_mass_in_amu": 23
        }
        return self.therm_spec

    def sput(self):
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
        return self.sput_spec

    # Particle emission position
    # ---------------------
    # Longitude and latitude distributions may be changed inside the 'create_particle' function.


def init3():
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

    # labels = ["Sun", "Jupiter", "Io"]
    # sim.add(labels)      # Note: Takes current position in the solar system. Therefore more useful to define objects manually in the following.
    sim.add(m=1.988e30, hash="sun")
    sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=sim.particles["sun"], hash="planet")  # Omega=1.753, omega=4.78
    sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles["planet"], hash="moon")
    sim.N_active = 3
    sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    sim.particles["planet"].r = 69911000
    sim.particles["moon"].r = 1821600


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

