from src.scheduler import SerpensScheduler
from src.species import Species


# Create a new scheduler instance
ssch = SerpensScheduler()

# ===== Example 1: Sodium from Europa =====
# Schedule a simulation of sodium particles from Europa (Jupiter's moon)
ssch.schedule(
    # Unique identifier for this simulation
    description="55-Amorph-Simulation",
    # Use the Jupiter-Europa system
    celestial_name='55-Cnc-e',
    # Specify that the moon (Europa) is the source object
    source_object='planet',
    # Define sodium as the species to simulate
    species=[
        Species(
            'SO2',  # Element symbol
            description='55 Amorphous Quartz',  # Description for plots and output
            n_th=0,  # Number of thermal particles (0 = disabled)
            n_sp=200,  # Number of sputtered particles per spawn
            mass_per_sec=5e6,  # Mass production rate (kg/s)
            model_smyth_v_b=1000,  # Bulk velocity parameter (m/s)
            model_smyth_v_M=20000,  # Maximum velocity parameter (m/s)
            lifetime=7874.5,
            beta=0.15
        )
    ],
    # Set integration parameters
    int_spec={
        "r_max": 6  # Maximum distance in units of semi-major axis
    }
)

# ===== Example 2: Sodium from Io =====
# Schedule a simulation of sodium particles from Io (Jupiter's moon)
ssch.schedule(
    description="55-Alpha-Simulation",
    celestial_name='55-Cnc-e',
    source_object='planet',
    species=[
        Species(
            'SO2',
            description='55 Alpha Quartz',
            n_th=0,
            n_sp=200,
            mass_per_sec=5e6,
            model_smyth_v_b=1000,
            model_smyth_v_M=20000,
            lifetime=39821,
            beta=0.15
        )
    ],
    int_spec={"r_max": 6}
)

# ===== Run all scheduled simulations =====
# This will execute both simulations sequentially
print("\nRunning all scheduled simulations...")
ssch.run(
    orbits=15,  # Run for 1 orbit of the source object
    spawns=375,  # Create particles 20 times during the simulation
    verbose=False  # Show detailed progress information
)