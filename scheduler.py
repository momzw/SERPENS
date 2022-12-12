import os as os
import shutil
from serpens_simulation import SerpensSimulation
from init import Parameters, Species


class NewParams:
# TODO: As subclass

    def __init__(self, species=None, objects=None, moon=None):
        self.species = species
        self.object_instructions = objects
        self.moon = moon
        Parameters.reset()

    def __call__(self):
        if self.species is not None:
            Parameters.modify_species(*self.species)

        if self.moon is not None:
            Parameters.modify_objects(moon=self.moon)

        if isinstance(self.object_instructions, dict):
            for k1, v1 in self.object_instructions.items():
                if isinstance(v1, dict) and k1 in Parameters.celest:
                    for k2, v2 in v1.items():
                        if k2 == "source" and v2 is True:
                            Parameters.modify_objects(object=k1, as_new_source=True)
                        if k2 == "m":
                            message = "can implement mass change etc. here"


        return Parameters()


if __name__ == "__main__":
    print("Scheduler called.")


    simulations = {
        "1": NewParams(),
        "2": NewParams(species=[Species('H2', description='H2--6.69kg/s--1200m/s', n_th=0, n_sp=1000, mass_per_sec=6.69, model_smyth_v_b=1200, model_smyth_v_M=50000),
                                Species('O2')],
                       objects={"add1": {"source": False}},
                       moon="default")
    }


    for k, v in simulations.items():
        v()
        sim = SerpensSimulation()
        sim.advance(Parameters.int_spec["num_sim_advances"])

        path = f'schedule_archive/simulation{k}'
        os.makedirs(path)
        shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
        shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/{path}")
        shutil.copy2(f'{os.getcwd()}/Parameters.txt', f"{os.getcwd()}/{path}")

        del sim

    print("=============================")
    print("COMPLETED ALL SIMULATIONS")
    print("=============================")















