import os as os
import shutil
from serpens_simulation import SerpensSimulation
from init import Parameters, Species


class NewParams:
    # TODO: As subclass
    """
    This class allows for the change of simulation parameters.

    Possible changes
    ----------------
    species : list
        New species to analyze.
        Signature:  Species(name: str, n_th: int, n_sp: int, mass_per_sec: float,
                            duplicate: int, beta: float, lifetime: float, n_e: float, sput_spec: dict)
            If attribute not passed to species, it is either set to 'None' or to default value.

    objects: dict
        Manipulate celestial objects.
        If passed 'objects: dict', a new source can be defined and object properties can be changed
        (see rebound.Particle properties)
            Example: {'Io': {'source': True, 'm': 1e23, 'a': 3e9}, 'Ganymede': {...}}

    moon: bool
        Change between moon and planet systems (e.g. Jovian to 55 Cnc)
        NOTE: It is currently only possible to change the chem. network via lifetime between systems.

    int_spec: dict
        update integration parameters. Only the specific dict parameter can be passed.

    therm_spec: dict
        update thermal evaporation parameters. Only the specific dict parameter can be passed.

    celest_set: int
        Change celestial object set according to objects.py.
        NOTE: Only works if 'moon: bool' is also defined.

    """

    def __init__(self, species=None, objects=None, moon=None, int_spec=None, therm_spec=None, celest_set=1):
        self.species = species
        self.object_instructions = objects
        self.moon = moon
        self.celest_set = celest_set
        self.int_spec = int_spec
        self.therm_spec = therm_spec
        Parameters.reset()

        if celest_set != 1 and moon is None:
            print("Please state if there is a moon for a new celestial body set using bool: 'moon'.")

    def __call__(self):
        Parameters()
        if self.species is not None:
            Parameters.modify_species(*self.species)

        if self.moon is not None:
            Parameters.modify_objects(moon=self.moon, set=self.celest_set)

        if isinstance(self.object_instructions, dict):
            for k1, v1 in self.object_instructions.items():
                if isinstance(v1, dict) and k1 in Parameters.celest:
                    source = v1.pop('source', False)
                    # False if v1 now empty:
                    if v1:
                        Parameters.modify_objects(object=k1, as_new_source=source, new_properties=v1)
                    else:
                        Parameters.modify_objects(object=k1, as_new_source=source)

        if self.therm_spec is not None or self.int_spec is not None:
            Parameters.modify_spec(self.int_spec, self.therm_spec)

        return


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
        "HD-189733-SuperEarth": NewParams(species=[
            Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
                    lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            celest_set=3),
        "HD-189733-SuperIo": NewParams(species=[
            Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
                    lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
            celest_set=3),
        "HD-189733-SuperEnce": NewParams(species=[
            Species('Na', description='Na--11576kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=11576,
                    lifetime=16.9 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 2.3e20, 'r': 250e3}},
            celest_set=3),
        "HD-209458-SuperEarth": NewParams(species=[
            Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
                    lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            celest_set=4),
        "HD-209458-SuperIo": NewParams(species=[
            Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
                    lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
            celest_set=4),
        "HD-209458-SuperEnce": NewParams(species=[
            Species('Na', description='Na--10233kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=10233,
                    lifetime=5.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 2.3e20, 'r': 250e3}},
            celest_set=4),
        "HAT-P-1-SuperEarth": NewParams(species=[
            Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
                    lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            celest_set=5),
        "HAT-P-1-SuperIo": NewParams(species=[
            Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
                    lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 8.8e22, 'r': 1820e3}},
            celest_set=5),
        "HAT-P-1-SuperEnce": NewParams(species=[
            Species('Na', description='Na--206909kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=206909,
                    lifetime=8.7 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)],
            moon=True,
            objects={'moon': {'m': 2.3e20, 'r': 250e3}},
            celest_set=5),
    }

    for k, v in simulations.items():
        v()
        sim = SerpensSimulation()
        sim.advance(Parameters.int_spec["num_sim_advances"])

        path = f'schedule_archive/simulation-{k}'

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/{path}")
        shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/{path}")
        shutil.copy2(f'{os.getcwd()}/Parameters.txt', f"{os.getcwd()}/{path}")
        shutil.copy2(f'{os.getcwd()}/Parameters.pickle', f"{os.getcwd()}/{path}")

        del sim

    print("=============================")
    print("COMPLETED ALL SIMULATIONS")
    print("=============================")















