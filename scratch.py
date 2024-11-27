from serpens_analyzer import SerpensAnalyzer
from serpens_simulation import SerpensSimulation
from src.species import Species
import matplotlib


if __name__ == "__main__":

    # "Empty" Jupiter simulation
    ssim = SerpensSimulation(system="Jupiter")

    # Io
    ssim.add(**{"m": 8.8e+22, "a": 421700000.0, "e": 0.0041, "r": 1821600, "primary": "planet", "source": True},
             species=Species('Na', n_th=0, n_sp=80,
                             mass_per_sec=10**4.8, model_smyth_v_b=0.95*1000,
                             model_smyth_v_M=15.24*1000, lifetime=4*60, beta=0)
             )

    # Europa
    ssim.add(**{"m": 4.799e+22, "a": 670900000.0, "e": 0.009, "inc": 0.0082, "r": 1560800, "primary": "planet", "source": True},
             species=Species('H2', n_th=0, n_sp=80,
                             mass_per_sec=10 ** 4.8, model_smyth_v_b=0.95 * 1000,
                             model_smyth_v_M=15.24 * 1000, lifetime=4 * 60, beta=0)
             )

    # Run
    ssim.advance(5, verbose=True)

    # Analyze
    cmap1 = matplotlib.colormaps["afmhot"]
    cmap2 = matplotlib.colormaps["cividis"]

    sa = SerpensAnalyzer(save_output=False, reference_system="source0")
    sa.plot_planar(timestep=4, d=2, scatter=True, triplot=True, show=True, trialpha=1,
                   lim=20, celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
                   colormap=[cmap1, cmap2], show_source=True, single_plot=True)

    #sa.plot_lineofsight(timestep=14, show=True, show_primary=False, show_moon=True, lim=8,
    #       celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], scatter=True,
    #       colormap=matplotlib.colormaps["afmhot"], single_plot=False)

    #sa.plot3d(14, log_cutoff=2)
