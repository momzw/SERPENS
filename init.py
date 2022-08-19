import rebound


class Simulation_Parameters:
    def __init__(self, name):
        self.var = {}
        self.name = name
    def add_var(self, key, value):
        self.var[key] = value
    def get_var(self, key):
        return self.var[key]

# Integration specifics
# ---------------------
# NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
sim_advance = 1/8  # Fraction of orbital period on each advance. When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
num_sim_advances = 20  # Number of times the simulation advances.
stop_at_steady_state = True
max_num_of_generation_advances = gen_max = None  # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.

# Generating particles
# ---------------------
num_thermal_per_advance = n_th = 0  # Number of particles created by thermal evap each integration advance.
num_sputter_per_advance = n_sp = 2000  # Number of particles created by sputtering each integration advance.
r_max = 1.8  # Maximal radial distance in units of moon semi-major axis. Particles beyond get removed from simulation.

# Thermal evaporation parameters
# ---------------------
Io_temp_max = 130
Io_temp_min = 90
spherical_symm_ejection = False
part_mass_in_amu = 23


int_spec = Simulation_Parameters('integration specifics')
gen_spec = Simulation_Parameters('generation specifics')
therm_spec = Simulation_Parameters('thermal evaporation specifics')
sput_spec = Simulation_Parameters('sputtering specifics')

int_spec.add_var("sim_advance", sim_advance)
int_spec.add_var("num_sim_advances", num_sim_advances)
int_spec.add_var("stop_at_steady_state", stop_at_steady_state)
int_spec.add_var("gen_max", gen_max)

gen_spec.add_var("n_th", n_th)
gen_spec.add_var("n_sp", n_sp)
gen_spec.add_var("r_max", r_max)

therm_spec.add_var("Io_temp_max", Io_temp_max)
therm_spec.add_var("Io_temp_min", Io_temp_min)
therm_spec.add_var("spherical_symm_ejection", spherical_symm_ejection)
therm_spec.add_var("part_mass_in_amu", part_mass_in_amu)


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
    sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=sim.particles[0], hash="planet")  # Omega=1.753, omega=4.78
    sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles[1], hash="moon")
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

    return sim