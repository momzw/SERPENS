import os as os
import shutil
import dill
from serpens_simulation import SerpensSimulation
from init import Parameters, NewParams
from species import Species



if __name__ == "__main__":

    print("Scheduler called.")

    # See NewParams class for possible simulation changes.
    simulations = {
        #"WASP-49-SuperEarth": NewParams(species=[Species('Na', description='Na--125285kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=125285, lifetime=4*60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #                                moon=True,
        #                                celest_set=2),
        #"WASP-49-SuperIo": NewParams(species=[Species('Na', description='Na--125285kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=125285, lifetime=4*60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #                             moon=True,
        #                             objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
        #                             celest_set=2),
        #"WASP-49-SuperEnce": NewParams(species=[Species('Na', description='Na--125285kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=125285, lifetime=4*60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #                               moon=True,
        #                               objects={'moon': {'m': 2.3e20, 'r': 250e3}},
        #                               celest_set=2),
        #"HD-189733-ExoEarth": NewParams(
        #    species=[Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
        #            lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    celest_set=3),
        #"HD-189733-ExoIo": NewParams(species=[
        #    Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
        #            lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
        #    celest_set=3),
        #"HD-189733-ExoEnce": NewParams(species=[
        #    Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
        #            lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 2.3e20, 'r': 250e3}},
        #    celest_set=3),
        #"HD-209458-ExoEarth": NewParams(species=[
        #    Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
        #            lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    celest_set=4),
        #"HD-209458-ExoIo": NewParams(species=[
        #    Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
        #            lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
        #    celest_set=4),
        #"HD-209458-ExoEnce": NewParams(species=[
        #    Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
        #            lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 2.3e20, 'r': 250e3}},
        #    celest_set=4),
        #"HAT-P-1-ExoEarth": NewParams(species=[
        #    Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
        #            lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    celest_set=5),
        #"HAT-P-1-ExoIo": NewParams(species=[
        #    Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
        #            lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
        #    celest_set=5),
        #"HAT-P-1-ExoEnce": NewParams(species=[
        #    Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
        #            lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
        #    moon=True,
        #    objects={'moon': {'m': 2.3e20, 'r': 250e3}},
        #    celest_set=5),
        "WASP-39-ExoIo": NewParams(
            species=[Species('Na', description='Na--2.6e12kg/s--2.5-30km/s--tau6.7min', n_th=0, n_sp=1000, mass_per_sec=2.6e12,
                             lifetime=6.7 * 360, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
            celest_set=6),
        "WASP-39-ExoEnce": NewParams(species=[
           Species('Na', description='Na--2.6e12kg/s--2.5-30km/s--tau6.7min', n_th=0, n_sp=1000, mass_per_sec=2.6e12,
                   lifetime=6.7 * 360, model_smyth_v_b=2500, model_smyth_v_M=30000)],
           moon=True,
           objects={'moon': {'m': 2.3e20, 'r': 250e3}},
           celest_set=6),
        "WASP-39-ExoEarth": NewParams(species=[
           Species('Na', description='Na--2.6e12kg/s--2.5-30km/s--tau6.7min', n_th=0, n_sp=1000, mass_per_sec=2.6e12,
                   lifetime=6.7 * 360, model_smyth_v_b=2500, model_smyth_v_M=30000)],
           moon=True,
           celest_set=6),
    }

    for k, v in simulations.items():
        v()
        with open("Parameters.pkl", 'wb') as f:
            dill.dump(v, f, protocol=dill.HIGHEST_PROTOCOL)
        sim = SerpensSimulation()
        sim.advance(Parameters.int_spec["num_sim_advances"])

        path = f'schedule_archive/simulation-{k}'

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
        shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/{path}")
        shutil.copy2(f'{os.getcwd()}/Parameters.txt', f"{os.getcwd()}/{path}")
        shutil.copy2(f'{os.getcwd()}/Parameters.pkl', f"{os.getcwd()}/{path}")

        del sim

    print("=============================")
    print("COMPLETED ALL SIMULATIONS")
    print("=============================")















