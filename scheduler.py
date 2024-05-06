import os as os
import shutil
import dill
from serpens_simulation import SerpensSimulation
from src.parameters import Parameters, NewParams
from src.species import Species


class SerpensScheduler:
    """
    The scheduler allows for queueing multiple simulations to run.
    Uses the 'NewParams' class to set each simulation's parameters and celestial objects.
    """
    sims = {}

    def schedule(self, description, species=None, objects=None, int_spec=None, therm_spec=None, celestial_name="Jupiter (Europa-Source)"):
        """
        Prepare a simulation to be run using the 'run' function. Populates the 'sims' dictionary with a set of
        parameters.
        This function does not start the simulation.

        Arguments
        ---------
        description : str
            Name/description of the simulation, which will be saved in the 'schedule_archive' subfolder.
        species : <class Species>   (can be a list)      (default: None)
            Species to be used in the simulation (single or multiple).
            Gets passed to NewParams.
        objects : dict      (default: None)
            Manipulation of existing objects from a system.
            A new source can be defined and object properties can be changed.
            (see rebound.Particle properties - https://rebound.readthedocs.io/en/latest/particles/)
            Gets passed to NewParams.
        int_spec : dict     (default: None)
            Update integration parameters. Only the specific dict parameter can be passed.
            Gets passed to NewParams.
        therm_spec : dict   (default: None)
            Update thermal evaporation parameters. Only the specific dict parameter can be passed.
            Gets passed to NewParams.
        celestial_name : str    (default: "Jupiter (Europa-Source)")
            Name of the celestial system to be used. Valid are entries from src/objects.txt
            Gets passed to NewParams.

        Examples
        --------
        *   ssch = SerpensScheduler()
            ssch.schedule("W49-ExoEarth-Na-physical-NORAD",
                          species=[Species('Na', description=r'WASP-49 exo-Earth $-$ Na (NO-RAD)', n_th=0, n_sp=800,
                                           mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
                                           model_smyth_v_M=15.24*1000, lifetime=4*60, beta=0)],
                          int_spec={"radiation_pressure_shield": False,
                                    "sim_advance": 1/120,
                                    "num_sim_advances": 360,
                                    "r_max": 8},
                          objects={'moon': {'m': 3.8e24, 'r': 6370000}},
                          celestial_name="WASP-49")
        *   ssch = SerpensScheduler()
            ssch.schedule("W39-ExoEnce Na NORAD",
                 species=[Species('Na', description=r'WASP-49 exo-Enceladus $-$ Na', n_th=0, n_sp=800,
                                  mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
                                  model_smyth_v_M=15.24*1000, lifetime=4*60, beta=3.19,
                                  shielded_lifetime=3*3600)],
                 int_spec={"radiation_pressure_shield": True,
                           "sim_advance": 1/120,
                           "num_sim_advances": 360},
                 objects={'moon': {'m': 2.3e20, 'r': 250000}},
                 celest_name="WASP-39")
        """
        if isinstance(description, str):
            self.sims[description] = NewParams(species=species, objects=objects, int_spec=int_spec, therm_spec=therm_spec, celestial_name=celestial_name)
        else:
            print("Please pass a string to the scheduler as simulation description.")

    def run(self, *args, **kwargs):
        """
        Runs and saves all simulations previously set by the 'schedule' function.

        Keyword Arguments
        -----------------
        save_freq : int         (default: 1)
            Save frequency of the simulation. A value of 5 means that the simulation gets saved every 5th advance.
        sim_advances : int      (default: Value from Parameters.int_spec)
            Number of simulation advances done.
        """

        print("Starting scheduled simulations.")
        save_freq = kwargs.get("save_freq", 1)
        for k, v in self.sims.items():
            v()
            num_advances = kwargs.get("sim_advances", Parameters.int_spec["num_sim_advances"])
            with open("Parameters.pickle", 'wb') as f:
                dill.dump(v, f, protocol=dill.HIGHEST_PROTOCOL)
            #sim = SerpensSimulation(filename, snapshot)
            sim = SerpensSimulation()
            sim.advance(num_advances, save_freq=save_freq)

            path = f"schedule_archive/simulation-{k}"

            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
            shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/{path}")
            shutil.copy2(f'{os.getcwd()}/Parameters.txt', f"{os.getcwd()}/{path}")
            shutil.copy2(f'{os.getcwd()}/Parameters.pkl', f"{os.getcwd()}/{path}")

            del sim

            Parameters.reset()

        print("=============================")
        print("COMPLETED ALL SIMULATIONS")
        print("=============================")



if __name__ == "__main__":
    ssch = SerpensScheduler()

    ssch.schedule("Test1",
                  celestial_name='Jupiter (Europa-Source)',
                  species=[Species('Na', description=r'Test1 $-$ Na', n_th=0, n_sp=300,
                                   mass_per_sec=6.69, model_smyth_v_b=1200,
                                   model_smyth_v_M=6*1000)],
                  int_spec={"sim_advance": 1/60,
                            "num_sim_advances": 20,
                            "r_max": 16}
                  )

    ssch.schedule("Test2",
                  celestial_name='Jupiter (Io-Source)',
                  species=[Species('Na', description=r'Test2 $-$ Na', n_th=0, n_sp=300,
                                   mass_per_sec=6.69, model_smyth_v_b=1200,
                                   model_smyth_v_M=6*1000)],
                  int_spec={"sim_advance": 1/60,
                            "num_sim_advances": 20,
                            "r_max": 16}
                  )

    ssch.run(save_freq=2)
