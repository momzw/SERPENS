from serpens_analyzer import SerpensAnalyzer, PhaseCurve
from scheduler import SerpensScheduler
import matplotlib

#ssch = SerpensScheduler()
#ssch.schedule("Test-Sim", celest_name="WASP-49")
#ssch.run(save_freq=1)

sa = SerpensAnalyzer(save_output=False, reference_system="geocentric")
sa.top_down(timestep=4, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1, lim=20,
            celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
            colormap=matplotlib.colormaps["afmhot"], show_moon=True, lvlmin=-10, lvlmax=15)

#sa.los(timestep=231, show=True, show_planet=False, show_moon=False, lim=4,
#       celest_colors=['yellow', 'sandybrown', 'yellow', 'gainsboro', 'tan', 'grey'], scatter=True, colormesh=False,
#       colormap=plt.cm.afmhot)

#sa.plot3d(121, log_cutoff=-5)
#sa.transit_curve('simulation-W69-ExoIo-Na-physical-HV')

#pc = PhaseCurve()
#pc.plot_curve_external('HD-189733', title="HD189", lifetime="photo", part_dens=True, column_dens=True, savefig=True)

#sa.phase_curve(load_path=[#'simulation-W17-ExoEarth-Na-physical-HV',
#                          #'simulation-W17-ExoIo-Na-physical-HV',
#                          #'simulation-W17-ExoEnce-Na-physical-HV',
#                          'simulation-W69-ExoEarth-Na-3h-HV',
#                          'simulation-W69-ExoIo-Na-3h-HV',
#                          'simulation-W69-ExoEnce-Na-3h-HV',
#                          ],
#               fig=True, part_dens=True, column_dens=False, title=r'WASP-69 b - 3h', savefig=False)
