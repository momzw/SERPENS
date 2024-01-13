from serpens_analyzer import SerpensAnalyzer
from serpens_simulation import SerpensSimulation
import matplotlib

ssim = SerpensSimulation(system="HD-189733")
ssim.add(m=8.8e+22, a=267868894.98, r=1820000.0, primary="planet", source=True)
#ssim.add(m=8.8e+22, a=200000000.98, r=1820000.0, primary="planet", source=True)
ssim.advance(40, verbose=False)

sa = SerpensAnalyzer(save_output=False, reference_system="source0")
sa.plot_planar(timestep=35, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1,
               lim=8, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
               colormap=matplotlib.colormaps["afmhot"], show_source=True)

#sa.plot_lineofsight(timestep=14, show=True, show_primary=False, show_moon=True, lim=8,
#       celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], scatter=True, colormesh=False,
#       colormap=matplotlib.colormaps["afmhot"])

#sa.plot3d(14, log_cutoff=2)