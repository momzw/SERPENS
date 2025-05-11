import os as os
import shutil
from src.serpens_simulation import SerpensSimulation
from src.parameters import GLOBAL_PARAMETERS, initialize_global_defaults
from src.species import Species


class SerpensScheduler:
    """
    The SerpensScheduler allows for queueing and running multiple SERPENS simulations sequentially.

    This class provides a convenient way to set up and run multiple simulations with different
    parameters, celestial systems, and species. It uses the GLOBAL_PARAMETERS to configure each
    simulation's settings and manages the creation, execution, and archiving of simulation results.

    Each simulation is configured using the 'schedule' method and then all scheduled simulations
    are executed sequentially using the 'run' method. Results are automatically saved to the
    'schedule_archive' directory with subdirectories named after each simulation's description.

    Attributes:
        sims (dict): Dictionary storing all scheduled simulations with their configurations.
    """
    sims = {}

    def schedule(self, description, species=None, objects=None, int_spec=None, therm_spec=None, celestial_name="Jupiter (Europa-Source)", source_object="moon"):
        """
        Prepare a simulation to be run using the 'run' function.

        This method configures a simulation with the specified parameters and adds it to the
        scheduler's queue. The simulation is not started immediately but will be executed when
        the 'run' method is called. Each simulation is identified by its description, which is
        also used as part of the directory name when saving results.

        Parameters
        ----------
        description : str
            Name/description of the simulation, which will be saved in the 'schedule_archive' subfolder.
        species : Species or list of Species (default: None)
            Species to be used in the simulation. Can be a single Species instance or a list of Species.
        objects : dict (default: None)
            Modifications to existing objects in the celestial system. The keys are object names,
            and the values are dictionaries of properties to modify.
            (see rebound.Particle properties - https://rebound.readthedocs.io/en/latest/particles/)
        int_spec : dict (default: None)
            Integration parameters to update. Common parameters include:
            - r_max: Maximum distance (in units of semi-major axis) before particles are removed
            - radiation_pressure_shield: Whether to shield particles from radiation pressure
        therm_spec : dict (default: None)
            Thermal evaporation parameters to update.
        celestial_name : str (default: "Jupiter (Europa-Source)")
            Name of the celestial system to use. Must be a valid entry from resources/objects.json.
        source_object : str (default: "moon")
            Name of the object to use as a source for the species.

        Examples
        --------
        Example 1: Simulating sodium particles from an Earth-like exoplanet

            ssch = SerpensScheduler()
            ssch.schedule("ExoEarth-Na-Simulation",
                          species=[Species('Na', description='Exo-Earth Na', n_th=0, n_sp=800,
                                           mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
                                           model_smyth_v_M=15.24*1000, lifetime=4*60, beta=0)],
                          int_spec={"radiation_pressure_shield": False, "r_max": 8},
                          objects={'moon': {'m': 3.8e24, 'r': 6370000}},
                          celestial_name="WASP-49",
                          source_object="moon")

        Example 2: Simulating sodium particles from an Enceladus-like exomoon

            ssch = SerpensScheduler()
            ssch.schedule("ExoEnceladus-Na-Simulation",
                          species=[Species('Na', description='Exo-Enceladus Na', n_th=0, n_sp=800,
                                           mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
                                           model_smyth_v_M=15.24*1000, lifetime=4*60, beta=3.19,
                                           shielded_lifetime=3*3600)],
                          int_spec={"radiation_pressure_shield": True, "r_max": 10},
                          objects={'moon': {'m': 2.3e20, 'r': 250000}},
                          celestial_name="WASP-39",
                          source_object="moon")
        """
        if isinstance(description, str):
            # Store simulation configuration
            self.sims[description] = {
                'species': species,
                'objects': objects,
                'int_spec': int_spec,
                'therm_spec': therm_spec,
                'celestial_name': celestial_name,
                'source_object': source_object
            }
        else:
            print("Please pass a string to the scheduler as simulation description.")

    def run(self, *args, **kwargs):
        """
        Run all scheduled simulations sequentially and save their results.

        This method executes all simulations that have been configured using the 'schedule' method.
        For each simulation, it:
        1. Creates a new SerpensSimulation instance with the specified celestial system
        2. Applies all configuration parameters (integration, thermal, object modifications)
        3. Sets up the source object and species
        4. Runs the simulation for the specified duration
        5. Saves the results to a subdirectory in 'schedule_archive'
        6. Resets the global parameters for the next simulation

        After all simulations are complete, a success message is displayed.

        Parameters
        ----------
        hours : float (default: None)
            Number of hours to run each simulation. If None, this parameter is not used.
        days : float (default: None)
            Number of days to run each simulation. If None, this parameter is not used.
        orbits : float (default: 1)
            Number of orbits to run each simulation. Default is 1 if no other time parameter is specified.
        spawns : int (default: None)
            Number of spawning events during each simulation. If None, only one spawning event occurs.
        verbose : bool (default: False)
            Whether to print detailed output during simulation execution.
        save_freq : int (default: 1)
            Legacy parameter, maintained for backward compatibility.

        Notes
        -----
        At least one of hours, days, or orbits should be specified to determine the simulation duration.
        If multiple time parameters are provided, they are added together.
        """

        print("Starting scheduled simulations.")
        save_freq = kwargs.get("save_freq", 1)
        hours = kwargs.get("hours", None)
        days = kwargs.get("days", None)
        orbits = kwargs.get("orbits", 1)  # Default to 1 orbit if nothing else is specified
        spawns = kwargs.get("spawns", None)
        verbose = kwargs.get("verbose", False)

        for description, config in self.sims.items():
            # Create a new simulation with the specified celestial system
            sim = SerpensSimulation(system=config['celestial_name'])

            # Apply integration parameters if specified
            if config['int_spec']:
                for key, value in config['int_spec'].items():
                    GLOBAL_PARAMETERS.set(key, value)

            # Apply thermal evaporation parameters if specified
            if config['therm_spec']:
                for key, value in config['therm_spec'].items():
                    GLOBAL_PARAMETERS.set(key, value)

            # Apply object modifications if specified
            if config['objects']:
                for obj_name, obj_props in config['objects'].items():
                    # This assumes the object already exists in the simulation
                    # and we're just modifying its properties
                    for prop, value in obj_props.items():
                        sim.particles[obj_name].params[prop] = value

            # Define the source object and species
            sim.object_to_source(
                config['source_object'],
                species=config['species']
            )

            # Run the simulation
            sim.advance(
                hours=hours,
                days=days,
                orbits=orbits,
                spawns=spawns,
                verbose=verbose
            )

            # Save the simulation results
            path = f"schedule_archive/simulation-{description}"

            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            # Copy the simulation files to the archive
            shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
            shutil.copy2(f"{os.getcwd()}/rebx.bin", f"{os.getcwd()}/{path}")
            shutil.copy2(f"{os.getcwd()}/parameters.pkl", f"{os.getcwd()}/{path}")
            shutil.copy2(f"{os.getcwd()}/particle_params.h5", f"{os.getcwd()}/{path}")
            if os.path.exists(f"{os.getcwd()}/source_parameters.pkl"):
                shutil.copy2(f"{os.getcwd()}/source_parameters.pkl", f"{os.getcwd()}/{path}")

            # Clean up
            del sim

            # Reset global parameters for the next simulation
            GLOBAL_PARAMETERS.reset()
            initialize_global_defaults(GLOBAL_PARAMETERS)

        print("=============================")
        print("COMPLETED ALL SIMULATIONS")
        print("=============================")



if __name__ == "__main__":
    """
    Example usage of the SerpensScheduler class.

    This example demonstrates how to:
    1. Create a scheduler
    2. Schedule multiple simulations with different parameters
    3. Run all simulations sequentially
    """
    # Create a new scheduler instance
    ssch = SerpensScheduler()

    # ===== Example 1: Sodium from Europa =====
    # Schedule a simulation of sodium particles from Europa (Jupiter's moon)
    ssch.schedule(
        # Unique identifier for this simulation
        description="Europa-Na-Simulation",
        # Use the Jupiter-Europa system
        celestial_name='Jupiter',
        # Specify that the moon (Europa) is the source object
        source_object='Europa',
        # Define sodium as the species to simulate
        species=[
            Species(
                'Na',                           # Element symbol
                description='Europa Sodium',    # Description for plots and output
                n_th=0,                         # Number of thermal particles (0 = disabled)
                n_sp=300,                       # Number of sputtered particles per spawn
                mass_per_sec=6.69,              # Mass production rate (kg/s)
                model_smyth_v_b=1200,           # Bulk velocity parameter (m/s)
                model_smyth_v_M=6*1000          # Maximum velocity parameter (m/s)
            )
        ],
        # Set integration parameters
        int_spec={
            "r_max": 16                         # Maximum distance in units of semi-major axis
        }
    )

    # ===== Example 2: Sodium from Io =====
    # Schedule a simulation of sodium particles from Io (Jupiter's moon)
    ssch.schedule(
        description="Io-Na-Simulation",
        celestial_name='Jupiter',
        source_object='Io',
        species=[
            Species(
                'Na', 
                description='Io Sodium',
                n_th=0,
                n_sp=300,
                mass_per_sec=6.69,
                model_smyth_v_b=1200,
                model_smyth_v_M=6*1000
            )
        ],
        int_spec={"r_max": 16}
    )

    # ===== Run all scheduled simulations =====
    # This will execute both simulations sequentially
    print("\nRunning all scheduled simulations...")
    ssch.run(
        orbits=1,       # Run for 1 orbit of the source object
        spawns=20,      # Create particles 20 times during the simulation
        verbose=True    # Show detailed progress information
    )

    print("\nSimulations complete. Results are saved in the 'schedule_archive' directory.")
    print("You can analyze the results using the SerpensAnalyzer class.")
