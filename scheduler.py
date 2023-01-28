import os as os
import shutil
import dill
from serpens_simulation import SerpensSimulation
from parameters import Parameters, NewParams
from species import Species


class SerpensScheduler:

    sims = {}

    def schedule(self, description, species=None, objects=None, moon=None, int_spec=None, therm_spec=None, celest_set=1):

        if isinstance(description, str):
            self.sims[description] = NewParams(species=species, objects=objects, moon=moon, int_spec=int_spec, therm_spec=therm_spec, celest_set=celest_set)
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
                print("You have passed additional arguments implying multiple scheduled simulation.")
                print("Please schedule only one simulation to append to the archive.")
                print("Returning...")
                return
            snapshot = -1

        print("Starting scheduled simulations.")
        for k, v in self.sims.items():
            v()
            num_advances = kwargs.get("sim_advances", Parameters.int_spec["num_sim_advances"])
            with open("Parameters.pickle", 'wb') as f:
                dill.dump(v, f, protocol=dill.HIGHEST_PROTOCOL)
            sim = SerpensSimulation(filename, snapshot)
            sim.advance(num_advances)

            path = f'schedule_archive/simulation-{k}'

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


# E.g.:
ssch = SerpensScheduler()

#######################################################################################################################
#ssch.schedule("Io-SO2",
#              species=[Species('SO2', description=r'SO2 $-$ 1000 kg/s $-$ .515-40km/s', n_th=0, n_sp=500,
#                               mass_per_sec=1000, model_smyth_v_b=515,
#                               model_smyth_v_M=40000)],
#              moon=True,
#              objects={"Io": {'source': True}},
#              celest_set=1)

#ssch.schedule("W39-Na+K",
#              species=[Species('Na', description=r'Na $-$ 63000 kg/s $-$ 1-40km/s', n_th=0, n_sp=500,
#                               mass_per_sec=63000, model_smyth_v_b=1000,
#                               model_smyth_v_M=40000),
#                       Species('K', description=r'K $-$ 63000 kg/s $-$ 1-40km/s', n_th=0, n_sp=500,
#                               mass_per_sec=63000, model_smyth_v_b=1000,
#                               model_smyth_v_M=40000)],
#              moon=True,
#              celest_set=6)
ssch.schedule("W49-K-photo",
              species=[Species('K', description=r'K $-$ 1-40km/s', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=64.37)],
              moon=True,
              celest_set=2,
              objects={"moon": {'a': 1.2 * 1.115 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/200, "num_sim_advances": 600})

ssch.schedule("W49-K-3h",
              species=[Species('K', description=r'K $-$ 1-40km/s - $\tau = 3$h', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=3 * 60 * 60)],
              moon=True,
              celest_set=2,
              objects={"moon": {'a': 1.2 * 1.115 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/40, "num_sim_advances": 400})

ssch.schedule("W69-K-photo",
              species=[Species('K', description=r'K $-$ 1-40km/s', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=88.63)],
              moon=True,
              celest_set=8,
              objects={"moon": {'a': 1.2 * 1.06 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/200, "num_sim_advances": 600})

ssch.schedule("W69-K-3h",
              species=[Species('K', description=r'K $-$ 1-40km/s - $\tau = 3$h', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=3 * 60 * 60)],
              moon=True,
              celest_set=8,
              objects={"moon": {'a': 1.2 * 1.06 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/40, "num_sim_advances": 400})


ssch.schedule("W49-Na-photo",
              species=[Species('Na', description=r'Na $-$ $N=10^{39}$ $-$ 1-40km/s', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=241)],
              moon=True,
              celest_set=2,
              objects={"moon": {'a': 2 * 1.115 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/100, "num_sim_advances": 600})

ssch.schedule("W49-Na-3h",
              species=[Species('Na', description=r'Na $-$ $N=10^{39}$ $-$ 1-40km/s - $\tau = 3$h', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=3 * 60 * 60)],
              moon=True,
              celest_set=2,
              objects={"moon": {'a': 2 * 1.115 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/40, "num_sim_advances": 400})

ssch.schedule("W69-Na-photo",
              species=[Species('Na', description=r'Na $-$ $N=5 \cdot 10^{36}$ $-$ 1-40km/s', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=332)],
              moon=True,
              celest_set=8,
              objects={"moon": {'a': 2 * 1.06 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/100, "num_sim_advances": 600})

ssch.schedule("W69-Na-3h",
              species=[Species('Na', description=r'Na $-$ $N=5 \cdot 10^{36}$ $-$ 1-40km/s - $\tau = 3$h', n_th=0, n_sp=1000,
                               mass_per_sec=63000, model_smyth_v_b=1000,
                               model_smyth_v_M=40000, lifetime=3 * 60 * 60)],
              moon=True,
              celest_set=8,
              objects={"moon": {'a': 2 * 1.06 * 69911e3}},
              int_spec={"r_max": 3, "sim_advance": 1/40, "num_sim_advances": 400})


ssch.run()












