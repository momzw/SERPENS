"""Small irradiated grain parameter study, not a calibrated survival model.

Run from the repository root: python -m testing.grain_parameter_study --quick
Each output directory must be new; existing simulation data is never overwritten.
"""
import argparse
import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.grains import GrainPopulation, OMITTED_PHYSICS
from src.parameters import GLOBAL_PARAMETERS
from src.serpens_analyzer import SerpensAnalyzer
from src.serpens_simulation import SerpensSimulation
from src.visualizing.visualize import Visualize


@contextmanager
def run_directory(path):
    original = Path.cwd()
    parameters = copy.deepcopy(GLOBAL_PARAMETERS.params)
    path.mkdir(parents=True, exist_ok=False)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)
        GLOBAL_PARAMETERS.params = parameters


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot encode {type(value)}')


def run_case(path, count, *, quick=False, backend='python'):
    steps, interval_s = (2, 300.) if quick else (8, 1800.)
    with run_directory(path):
        GLOBAL_PARAMETERS.update(dict(celest={}, species=[], all_species=[], grains={}, all_grains={},
                                      lorentz_enabled=False, fix_source_circular_orbit=False,
                                      grain_seed=1847, integration_threads=2, integration_max_dt=60.,
                                      gen_max=None, r_max=5.))
        simulation_class = SerpensSimulation
        if backend == 'c':
            from src.cerpens.cerpens_simulation import CerpensSimulation
            simulation_class = CerpensSimulation
        simulation = simulation_class()
        simulation.add(m=1.989e30, r=6.96e8, hash='star')
        simulation.add(m=1.898e27, r=7e7, a=0.2 * 1.495978707e11,
                       primary='star', hash='planet')
        simulation.add(m=8.8e22, r=1.8e6, a=4.217e8, primary='planet', hash='moon')
        simulation.move_to_com()
        radii = (0.3e-6, 1e-6, 3e-6)
        populations = [GrainPopulation(
            name=f'parameter grain {radius * 1e6:g} um', density_kg_m3=2000.,
            radius_min_m=radius, radius_max_m=radius, size_slope=3.5,
            tracers_per_injection=count, dust_mass_per_sec=1.,
            speed_ref_m_s=4000., radius_ref_m=1e-6, speed_exponent=0.5,
            speed_min_m_s=2000., speed_max_m_s=8000., launch_altitude_m=1e4,
            radiation_source='star', luminosity_w=3.828e26, q_pr=1., potential_v=0.,
        ) for radius in radii]
        simulation.object_to_source('moon', grains=populations)
        simulation.save_to_file('simdata/archive.bin', delete_file=True)
        for _ in range(steps):
            simulation.advance_single(time=interval_s, orbit_object='moon')
            simulation.serpens_iter += 1

        analyzer = SerpensAnalyzer(reference_system='moon')
        analyzer.load_timestep_data(-1)
        center = analyzer.sim.particles['planet'].xyz
        edges = np.linspace(0., 5. * 4.217e8, 41)
        budgets = {}
        profiles = {}
        for population_id, radius in enumerate(radii, start=1):
            budget = analyzer.grain_budget(-1, population_id=population_id)
            expected_mass = interval_s * steps
            if not np.isclose(budget['injected']['mass'], expected_mass, rtol=1e-12):
                raise AssertionError('Injected dust mass does not match the configured production')
            if not np.isclose(budget['retained']['mass'] + budget['removed']['mass'], expected_mass, rtol=1e-12):
                raise AssertionError('Retained and removed grain mass does not close the budget')
            if len(budget['unaccounted_hashes']):
                raise AssertionError('Unaccounted grain losses')
            budgets[str(population_id)] = budget
            profile = analyzer.grain_radial_profile(-1, edges, quantity='mass',
                                                   center=center, population_id=population_id)
            profiles[str(population_id)] = profile
            np.savetxt(f'radial_mass_{radius * 1e6:g}um.csv',
                       np.column_stack((profile['radii'], profile['integrated_weights'], profile['column_density'])),
                       delimiter=',', header='cylindrical_radius_m,annulus_mass_kg,column_mass_kg_m-2')

        with mpl.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            fig, ax = plt.subplots(figsize=(6, 4))
            for population_id, radius in enumerate(radii, start=1):
                profile = profiles[str(population_id)]
                ax.plot(profile['radii'] / 7e7, profile['column_density'], label=f'{radius * 1e6:g} um')
            ax.set(xlabel='Cylindrical radius / planet radius', ylabel='Column mass density [kg m^-2]',
                   title=f'Early grain distribution: {count} tracers/population/injection')
            ax.legend()
            fig.tight_layout()
            fig.savefig('radial_profiles.png', dpi=120)
            plt.close(fig)
            field = analyzer.grain_field(-1, quantity='area', d=3)
            visualizer = Visualize(analyzer.sim, 'moon', interactive=False, panel_count=1,
                                   shadow_polygon=False, planetstar_connection=False, lim=10, figsize=5)
            visualizer.add_grain_field(field)
            visualizer.fig.savefig('geometric_area_density.png', dpi=100, bbox_inches='tight')
            plt.close(visualizer.fig)

        summary = dict(backend=backend, seed=1847, tracers_per_injection=count, steps=steps,
                       interval_s=interval_s, integration_max_dt=60., populations=[p.to_dict() for p in populations],
                       budgets=budgets, omitted_physics=OMITTED_PHYSICS,
                       validity='Uncalibrated fixed-size, optically thin, collisionless parameter study; '
                                'survival and environmental drag have not been estimated.')
        with open('summary.json', 'w') as handle:
            json.dump(summary, handle, indent=2, default=json_value)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/grain-study', help='New output directory')
    parser.add_argument('--quick', action='store_true', help='Two short injections and counts 8/16')
    parser.add_argument('--backend', choices=('python', 'c'), default='python')
    args = parser.parse_args()
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    counts = (8, 16) if args.quick else (32, 128)
    results = [run_case(output / f'tracers_{count}', count, quick=args.quick, backend=args.backend)
               for count in counts]
    with (output / 'comparison.json').open('w') as handle:
        json.dump(results, handle, indent=2, default=json_value)
    print(f'Grain study saved to {output}; all injected/retained/removed mass budgets close.')


if __name__ == '__main__':
    main()