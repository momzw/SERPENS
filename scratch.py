from serpens_analyzer import SerpensAnalyzer
from serpens_simulation import SerpensSimulation
from src.species import Species
import matplotlib

#ssim = SerpensSimulation(system="HD-189733")
#ssim.add(m=8.8e+22, a=267868894.98, r=1820000.0, primary="planet", source=True, species=Species('H2', n_th=0, n_sp=80, mass_per_sec=10**4.8,
#                                                                                                model_smyth_v_b=0.95*1000, model_smyth_v_M=15.24*1000,
#                                                                                                lifetime=4*60, beta=0))
#ssim.add(m=8.8e+22, a=200000000.98, r=1820000.0, primary="planet", source=True)
#ssim.advance(100, verbose=False)

cmap1 = matplotlib.colormaps["afmhot"]
cmap2 = matplotlib.colormaps["cividis"]

sa = SerpensAnalyzer(save_output=False, reference_system="source0")
#sa.plot_planar(timestep=20, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1,
#               lim=8, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
#               colormap=[cmap1, cmap2], show_source=True, single_plot=False)

#sa.plot_lineofsight(timestep=14, show=True, show_primary=False, show_moon=True, lim=8,
#       celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], scatter=True, colormesh=False,
#       colormap=matplotlib.colormaps["afmhot"])

sa.plot3d(14, log_cutoff=2)
