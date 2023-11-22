from serpens_analyzer import SerpensAnalyzer, PhaseCurve
from serpens_simulation import SerpensSimulation
from scheduler import SerpensScheduler
from src.parameters import Parameters
import matplotlib


ssim = SerpensSimulation(system="HD-189733")
ssim.advance(15, verbose=False)

sa = SerpensAnalyzer(save_output=False, reference_system="geocentric")
sa.plot_planar(timestep=14, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1,
               lim=8, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
               colormap=matplotlib.colormaps["afmhot"], show_source=True)

#sa.plot_lineofsight(timestep=5, show=True, show_planet=True, show_moon=True, lim=8,
#       celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], scatter=True, colormesh=False,
#       colormap=matplotlib.colormaps["afmhot"])

#sa.plot3d(15, log_cutoff=2)

""" ############################################################# """


#from multi_simulations import SerpensMultiSimulation
#smulti = SerpensMultiSimulation()
#smulti.load_from_schedule_archive(('simulation-Test1', 'simulation-Test2'))
#smulti.plot_planar(timestep=1, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1,
#               lim=16, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
#               colormap=matplotlib.colormaps["afmhot"], show_source=True)

#sa.transit_curve('simulation-W69-ExoIo-Na-physical-HV')

#pc = PhaseCurve()
#pc.plot_curve_external('HD-189733', title="HD189", lifetime="photo", part_dens=True, column_dens=True, savefig=False)

#sa.phase_curve(load_path=[#'simulation-W17-ExoEarth-Na-physical-HV',
#                          #'simulation-W17-ExoIo-Na-physical-HV',
#                          #'simulation-W17-ExoEnce-Na-physical-HV',
#                          'simulation-W69-ExoEarth-Na-3h-HV',
#                          'simulation-W69-ExoIo-Na-3h-HV',
#                          'simulation-W69-ExoEnce-Na-3h-HV',
#                          ],
#               fig=True, part_dens=True, column_dens=False, title=r'WASP-69 b - 3h', savefig=False)
