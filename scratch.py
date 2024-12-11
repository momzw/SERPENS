from serpens_analyzer import SerpensAnalyzer
from serpens_simulation import SerpensSimulation
from src.species import Species
import matplotlib


if __name__ == "__main__":

    # "Empty" Jupiter simulation
    ssim = SerpensSimulation(system="Jupiter")

    # Add Io and Europa to the simulation (in the future this will also be possible using the NASA JPL Horizons database)
    ssim.add(m=8.8e+22, a=421700000.0, e=0.0041, r=1821600, primary="Jupiter", hash="Io")
    ssim.add(m=4.799e+22, a=670900000.0, e=0.009, inc=0.0082, r=1560800, primary="Jupiter", hash="Europa")

    # Declare Io as a source for Sodium
    ssim.object_to_source(
        "Io",
        species=Species(
            'Na',
            n_th=0, n_sp=80,
            mass_per_sec=10**4.8,
            model_smyth_v_b=0.95*1000,
            model_smyth_v_M=15.24*1000,
            lifetime=4*60,
            beta=0
        )
    )

    # Declare Europa as a source for Hydrogen
    ssim.object_to_source(
        "Europa",
        species=Species(
            'H2',
            n_th=0, n_sp=80,
            mass_per_sec=10 ** 4.8,
            model_smyth_v_b=0.95 * 1000,
            model_smyth_v_M=15.24 * 1000,
            lifetime=4 * 60,
            beta=0)
    )

    # Run the simulation for a short duration
    ssim.advance(3, verbose=True)

    # Create the SerpensAnalyzer object
    sa = SerpensAnalyzer(reference_system="Io")

    # Set up colormaps for Sodium and Hydrogen
    cmap1 = matplotlib.colormaps["afmhot"]
    cmap2 = matplotlib.colormaps["cividis"]

    # Plot the planar view of the simulation
    sa.plot_planar(
        timestep=4, d=2, scatter=True, triplot=True, show=True, trialpha=.6, lim=15,
        celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], colormap=[cmap1, cmap2],
        show_source=True, single_plot=True
    )

    # Plot the line of sight view of the simulation
    sa.plot_lineofsight(
        timestep=2, scatter=True, show=True, show_primary=True, show_moon=True, lim=15,
        celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'], colormap=[cmap1, cmap2],
        single_plot=False
    )

