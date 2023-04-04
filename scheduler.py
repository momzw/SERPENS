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


ssch = SerpensScheduler()

#######################################################################################################################

#ssch.schedule("W49-ExoEarth-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-49 exo-Io $-$ Na', n_th=0, n_sp=800,
#                               mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=15.24*1000, lifetime=4*60, beta=3.19,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=2)

#ssch.schedule("W49-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-49 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=15.24*1000, lifetime=3*3600, beta=3.19)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=2)

##############################################################################

#ssch.schedule("W39-ExoEnce-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-39 exo-Io $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**5.8, model_smyth_v_b=0.85*1000,
#                               model_smyth_v_M=11.86*1000, lifetime=6.7*60, beta=1.70,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=6)
#
#ssch.schedule("W39-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-39 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**5.8, model_smyth_v_b=0.85*1000,
#                               model_smyth_v_M=11.86*1000, lifetime=3*3600, beta=1.70)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=6)

#ssch.schedule("W39-ExoIo-SO2-physical-RAD-fast",
#              species=[Species('SO2', description=r'WASP-39 exo-Io $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**10, model_smyth_v_b=2000,
#                               model_smyth_v_M=30000, lifetime=2.47*60, beta=1.70,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=6)
#
#ssch.schedule("W39-ExoIo-SO2-3h-RAD-fast",
#              species=[Species('SO2', description=r'WASP-39 exo-Io $-$ SO2 ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**10, model_smyth_v_b=2000,
#                               model_smyth_v_M=30000, lifetime=3*3600, beta=1.70)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/100,
#                        "num_sim_advances": 400},
#              celest_set=6)
#
#ssch.schedule("W39-ExoIo-SO2-physical-fast",
#              species=[Species('SO2', description=r'WASP-39 exo-Io $-$ SO2', n_th=0, n_sp=500,
#                               mass_per_sec=10**10, model_smyth_v_b=2000,
#                               model_smyth_v_M=30000, lifetime=2.47*60, beta=0,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=6)
#
#ssch.schedule("W39-ExoIo-SO2-3h-fast",
#              species=[Species('SO2', description=r'WASP-39 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**10, model_smyth_v_b=2000,
#                               model_smyth_v_M=30000, lifetime=3*3600, beta=0)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/100,
#                        "num_sim_advances": 400},
#              celest_set=6)

#############################################################################

#ssch.schedule("W17-ExoEnce-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-17 exo-Io $-$ Na', n_th=0, n_sp=1000,
#                               mass_per_sec=10**4.4, model_smyth_v_b=1.06*1000,
#                               model_smyth_v_M=16.02*1000, lifetime=3.4*60, beta=5.37,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/150,
#                        "num_sim_advances": 375},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=7)

#ssch.schedule("W17-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-17 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.4, model_smyth_v_b=1.06*1000,
#                               model_smyth_v_M=16.02*1000, lifetime=3*3600, beta=5.37)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/150,
#                        "num_sim_advances": 375},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=7)

##############################################################################

#ssch.schedule("W69-ExoEnce-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-69 exo-Io $-$ Na', n_th=0, n_sp=800,
#                               mass_per_sec=10**6.6, model_smyth_v_b=0.79*1000,
#                               model_smyth_v_M=12.43*1000, lifetime=35.9*60, beta=1.27,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=8)
#

#ssch.schedule("W69-ExoEarth-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-69 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=800,
#                               mass_per_sec=10**6.6, model_smyth_v_b=0.79*1000,
#                               model_smyth_v_M=12.43*1000, lifetime=3*3600, beta=1.27)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=8)

##############################################################################

#ssch.schedule("HD189-ExoIo-Na-physical-UHV",
#              species=[Species('Na', description=r'HD-189733 exo-Io $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**5.5, model_smyth_v_b=10000, # 0.87*1000
#                               model_smyth_v_M=60000, lifetime=16.9*60, beta=2,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=3)

#ssch.schedule("HD189-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'HD-189733 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**5.5, model_smyth_v_b=0.87*1000,
#                               model_smyth_v_M=23.80*1000, lifetime=3*3600, beta=2)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=3)

##############################################################################

#ssch.schedule("HD209-ExoEnce-Na-physical-HV",
#              species=[Species('Na', description=r'HD-209458 exo-Io $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.3, model_smyth_v_b=0.97*1000,
#                               model_smyth_v_M=17.82*1000, lifetime=5.7*60, beta=4,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=4)

#ssch.schedule("HD209-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'HD-209458 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.3, model_smyth_v_b=0.97*1000,
#                               model_smyth_v_M=17.82*1000, lifetime=3*3600, beta=4)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=4)

##############################################################################

#ssch.schedule("HATP1-ExoEarth-Na-physical-HV",
#              species=[Species('Na', description=r'HAT-P-1 exo-Io $-$ Na', n_th=0, n_sp=800,
#                               mass_per_sec=10**4.3, model_smyth_v_b=0.93*1000,
#                               model_smyth_v_M=13.35*1000, lifetime=8.7*60, beta=2.62,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=5)
#
#ssch.schedule("HATP1-ExoEarth-Na-3h-HV",
#              species=[Species('Na', description=r'HAT-P-1 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=800,
#                               mass_per_sec=10**5.3, model_smyth_v_b=0.93*1000,
#                               model_smyth_v_M=13.35*1000, lifetime=3*3600, beta=2.62)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=5)

##############################################################################

#ssch.schedule("W96-ExoEarth-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Earth $-$ Na', n_th=0, n_sp=800,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=5.8*60, beta=2.41,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=9)
#
#ssch.schedule("W96-ExoEarth-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Earth $-$ Na ($\tau=3$h)', n_th=0, n_sp=800,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=3*3600, beta=2.41)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=9)

#ssch.schedule("W96-ExoIo-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Io $-$ Na', n_th=0, n_sp=800,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=5.8*60, beta=2.41,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=9)

#ssch.schedule("W96-ExoIo-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=3*3600, beta=2.41)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=9)
#
#ssch.schedule("W96-ExoEnce-Na-physical-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Enceladus $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=5.8*60, beta=2.41,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=9)
#
#ssch.schedule("W96-ExoEnce-Na-3h-HV",
#              species=[Species('Na', description=r'WASP-96 exo-Enceladus $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.5, model_smyth_v_b=0.9*1000,
#                               model_smyth_v_M=18.58*1000, lifetime=3*3600, beta=2.41)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 2.3e20, 'r': 250000}},
#              celest_set=9)

##############################################################################

#ssch.schedule("XO2N-ExoEarth-Na-physical-HV",
#              species=[Species('Na', description=r'XO2N exo-Earth $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=19.75*1000, lifetime=3.8*60, beta=2.58,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=10)
#
#ssch.schedule("XO2N-ExoEarth-Na-3h-HV",
#              species=[Species('Na', description=r'XO2N exo-Earth $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=19.75*1000, lifetime=3*3600, beta=2.58)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              objects={'moon': {'m': 3.8e24, 'r': 6370000}},
#              celest_set=10)
#
#ssch.schedule("XO2N-ExoIo-Na-physical-HV",
#              species=[Species('Na', description=r'XO2N exo-Io $-$ Na', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=19.75*1000, lifetime=3.8*60, beta=2.58,
#                               shielded_lifetime=3*3600)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=10)
#
#ssch.schedule("XO2N-ExoIo-Na-3h-HV",
#              species=[Species('Na', description=r'XO2N exo-Io $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
#                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
#                               model_smyth_v_M=19.75*1000, lifetime=3*3600, beta=2.58)],
#              moon=True,
#              int_spec={"radiation_pressure_shield": True,
#                        "sim_advance": 1/120,
#                        "num_sim_advances": 360},
#              celest_set=10)

ssch.schedule("XO2N-ExoEnce-Na-physical-HV",
              species=[Species('Na', description=r'XO2N exo-Enceladus $-$ Na', n_th=0, n_sp=500,
                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
                               model_smyth_v_M=19.75*1000, lifetime=3.8*60, beta=2.58,
                               shielded_lifetime=3*3600)],
              moon=True,
              int_spec={"radiation_pressure_shield": True,
                        "sim_advance": 1/120,
                        "num_sim_advances": 360},
              objects={'moon': {'m': 2.3e20, 'r': 250000}},
              celest_set=10)

ssch.schedule("XO2N-ExoEnce-Na-3h-HV",
              species=[Species('Na', description=r'XO2N exo-Enceladus $-$ Na ($\tau=3$h)', n_th=0, n_sp=500,
                               mass_per_sec=10**4.9, model_smyth_v_b=0.95*1000,
                               model_smyth_v_M=19.75*1000, lifetime=3*3600, beta=2.58)],
              moon=True,
              int_spec={"radiation_pressure_shield": True,
                        "sim_advance": 1/120,
                        "num_sim_advances": 360},
              objects={'moon': {'m': 2.3e20, 'r': 250000}},
              celest_set=10)

##############################################################################

ssch.run(save_freq=5)












