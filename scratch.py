from serpens_analyzer import SerpensAnalyzer
from serpens_simulation import SerpensSimulation
import matplotlib

ssim = SerpensSimulation(system="HD-189733")
ssim.advance(15, verbose=False)

sa = SerpensAnalyzer(save_output=False, reference_system="source0")
sa.plot_planar(timestep=14, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1,
               lim=8, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
               colormap=matplotlib.colormaps["afmhot"], show_source=True)

#sa.plot_lineofsight(timestep=14, show=True, show_primary=False, show_moon=True, lim=8,
#       celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], scatter=True, colormesh=False,
#       colormap=matplotlib.colormaps["afmhot"])

#sa.plot3d(14, log_cutoff=2)