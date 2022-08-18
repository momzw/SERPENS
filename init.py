import rebound

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