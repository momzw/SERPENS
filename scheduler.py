import os as os
import shutil
import dill
from serpens_simulation import SerpensSimulation
from src.parameters import Parameters, NewParams


class SerpensScheduler:

    sims = {}

    def schedule(self, description, species=None, objects=None, int_spec=None, therm_spec=None, celest_name="Jupiter"):

        if isinstance(description, str):
            self.sims[description] = NewParams(species=species, objects=objects, int_spec=int_spec, therm_spec=therm_spec, celestial_name=celest_name)
        else:
            print("Please pass a string to the scheduler as simulation description.")

    def run(self, *args, **kwargs):

        if len(self.sims) == 1:
            # Handle arguments
            filename = None
            if len(args) > 0:
                filename = args[0]
            if "filename" in kwargs:
                filename = kwargs["filename"]
            snapshot = -1
            if len(args) > 1:
                snapshot = args[1]
            if "snapshot" in kwargs:
                snapshot = kwargs["snapshot"]
        else:
            filename = None
            if len(args) > 0 or "filename" in kwargs:
                print("You have passed additional arguments implying multiple scheduled simulations.")
                print("Please schedule only one simulation to append to the archive.")
                print("Returning...")
                return
            snapshot = -1

        print("Starting scheduled simulations.")
        save_freq = kwargs.get("save_freq", 1)
        for k, v in self.sims.items():
            v()
            num_advances = kwargs.get("sim_advances", Parameters.int_spec["num_sim_advances"])
            with open("Parameters.pickle", 'wb') as f:
                dill.dump(v, f, protocol=dill.HIGHEST_PROTOCOL)
            sim = SerpensSimulation(filename, snapshot)
            sim.advance(num_advances, save_freq=save_freq)

            path = f"schedule_archive/simulation-{k}"

            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
            shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/{path}")
            shutil.copy2(f'{os.getcwd()}/Parameters.txt', f"{os.getcwd()}/{path}")
            shutil.copy2(f'{os.getcwd()}/Parameters.pickle', f"{os.getcwd()}/{path}")

            del sim

            Parameters.reset()

        print("=============================")
        print("COMPLETED ALL SIMULATIONS")
        print("=============================")


#ssch = SerpensScheduler()
#
#ssch.schedule("Europa-H2",
#              species=[Species('H2', description=r'Europa $-$ H2', n_th=0, n_sp=300,
#                               mass_per_sec=6.69, model_smyth_v_b=1200,
#                               model_smyth_v_M=40*1000)],
#              moon=True,
#              int_spec={"sim_advance": 1/100,
#                        "num_sim_advances": 500,
#                        "r_max": 4}
#              )
#
#ssch.run(save_freq=1)












