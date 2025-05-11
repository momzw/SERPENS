from src.serpens_analyzer import SerpensAnalyzer
from src.serpens_simulation import SerpensSimulation
from src.species import Species
import matplotlib


if __name__ == "__main__":

    ssim = SerpensSimulation(system="55-Cnc-e")
    
    # Declare Io as a source for Sodium
    ssim.object_to_source(
        "planet",
        species=Species(
            'SO2',
            n_th=0, n_sp=50,
            mass_per_sec=5e6,
            model_smyth_v_b=1000,
            model_smyth_v_M=20000,
            lifetime=7874.5, # 39821 (alpha quartz),  7874.5 (amorphous quartz)
            beta=0.15
        )
    )

    # Run the simulation for a short duration
    ssim.advance(orbits=2, spawns=20, verbose=False)

    # Create the SerpensAnalyzer object
    sa = SerpensAnalyzer(save_output=False)

    # Plot the planar view of the simulation
    sa.calculate_phasecurve('planet', orbits=1)
    sa.plot_phasecurve(filename='phase-test.csv', type='mean', particle_density=False)

    #sa.plot_planar(
    #    timestep=10, d=3, scatter=True, triplot=False, show=True, trialpha=.6, lim=15,
    #    celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
    #    show_source=True, single_plot=True, interactive=False, lvl_min=0, #lvl_max=10
    #)

    # Plot the line of sight view of the simulation
    #sa.plot_lineofsight(
    #    timestep=10, scatter=True, show=True, show_primary=True, show_moon=True, lim=15,
    #    celest_colors=['yellow', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
    #    interactive=False, lvl_min=4,# lvl_max=19
    #)


