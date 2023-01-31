import numpy as np
import shutil
import os
import pandas as pd
from serpens_analyzer import SerpensAnalyzer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
#plt.style.use('seaborn')


PATHS = ['simulation-HATP1-ExoIo-Na-3h',
         'simulation-HATP1-ExoIo-Na-physical',
         #'simulation-HD189-ExoIo-Na-physical',
         'simulation-HD189-ExoIo-Na-3h-HighVel',
         'simulation-HD189-ExoIo-Na-physical-HighVel',
         'simulation-W17-ExoIo-Na-3h',
         'simulation-W17-ExoIo-Na-3h-HighVel',
         'simulation-W17-ExoIo-Na-physical',
         'simulation-W17-ExoIo-Na-physical-HighVel',
         'simulation-W39-ExoIo-Na-3h',
         'simulation-W39-ExoIo-Na-3h-HighVel',
         'simulation-W39-ExoIo-Na-physical',
         'simulation-W39-ExoIo-Na-physical-HighVel',
         'simulation-W49-ExoIo-Na-3h',
         'simulation-W49-ExoIo-Na-3h-HighVel',
         'simulation-W49-ExoIo-Na-physical',
         'simulation-W49-ExoIo-Na-physical-HighVel',
         #'simulation-W39-ExoIo-SO2-physical',
         #'simulation-W39-ExoIo-SO2-physical-RAD',
         #'simulation-W39-ExoIo-SO2-3h',
         #'simulation-W39-ExoIo-SO2-3h-RAD',
         ]


def plot_run(path, top_down=True, LOS=False):

    print(f"Started {path[11:]}")

    print("\t copying ...")
    #shutil.copy(f'{os.getcwd()}/schedule_archive/{path}/archive.bin', f'{os.getcwd()}')
    #shutil.copy(f'{os.getcwd()}/schedule_archive/{path}/hash_library.pickle', f'{os.getcwd()}')
    #shutil.copy(f'{os.getcwd()}/schedule_archive/{path}/Parameters.pickle', f'{os.getcwd()}')
    #shutil.copy(f'{os.getcwd()}/schedule_archive/{path}/Parameters.txt', f'{os.getcwd()}')
    print("\t ... done!")

    sa = SerpensAnalyzer(save_output=True, folder_name=path[11:], reference_system="geocentric", r_cutoff=4)

    print("\t Creating plots ...")
    if top_down:
        sa.top_down(timestep=np.arange(11, len(sa.sa) - len(sa.sa) % 5 + 1, 5), d=3,
                    colormesh=False, scatter=True, triplot=False, show=False,
                    smoothing=.5, trialpha=.7, lim=4,
                    celest_colors=['yellow', 'sandybrown', 'yellow', 'yellow', 'green', 'green'],
                    colormap=plt.cm.get_cmap("afmhot"), lvlmin=0, lvlmax=15)
    if LOS:
        sa.los(timestep=np.arange(11, len(sa.sa) - len(sa.sa) % 5 + 1, 5),
               colormesh=False, scatter=True, show=False,
               show_planet=False, show_moon=False, lim=4,
               celest_colors=['yellow', 'sandybrown', 'yellow', 'yellow', 'green', 'green'],
               colormap=plt.cm.autumn, lvlmax=18, lvlmin=0)

    print("\t ... done!")
    print("____________________________________________")

    del sa

    return


def generate_phase_curves():
    for path in PATHS:
        print(f"Started {path[11:]}")

        print("\t copying ...")
        shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/archive.bin', f'{os.getcwd()}')
        shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/hash_library.pickle', f'{os.getcwd()}')
        shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/Parameters.pickle', f'{os.getcwd()}')
        shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/Parameters.txt', f'{os.getcwd()}')
        print("\t ... done!")

        sa = SerpensAnalyzer(save_output=False, reference_system="geocentric", r_cutoff=4)

        print("\t Calculating phase curve ...")
        sa.phase_curve(title=path[11:], fig=True, savefig=True, save_data=False, load_path=path)
        print("\t ... done!")
        print("____________________________________________")

        del sa


def read_pkl():

    #color1 = '#FFFFF0'
    #color2 = '#FFA500'
    color1 = matplotlib.colors.to_hex('ivory')
    color2 = matplotlib.colors.to_hex('darkorange')

    def hex_to_RGB(hex_str):
        """ #FFFFFF -> [255,255,255]"""
        # Pass 16 to the integer function for change of base
        return [int(hex_str[i:i + 2], 16) for i in range(1, 6, 2)]

    def get_color_gradient(c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        assert n > 1
        c1_rgb = np.array(hex_to_RGB(c1)) / 255
        c2_rgb = np.array(hex_to_RGB(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]

    phase_suparray = []
    los_mean_suparray = []
    dens_mean_suparray = []
    for path in PATHS:
        df = pd.read_pickle(f"./schedule_archive/phaseCurves/data/PhaseCurveData-{path[11:]}.pkl")
        phase_suparray.append(df["phases"].values)
        los_mean_suparray.append(df["mean_los"].values)
        dens_mean_suparray.append(df["mean_dens"].values)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    fig.suptitle(f"Phase-Density Curves", fontsize='x-large')

    for i, path in enumerate(PATHS):
        phases = phase_suparray[i]
        los_mean = los_mean_suparray[i]
        dens_mean = dens_mean_suparray[i]

        axs[0].plot(phases, dens_mean, label=f"{path[11:]}", color='orange')
        #axs[0].set_facecolor('black')
        #axs[0].scatter(phases, dens_mean)
        axs[1].plot(phases, los_mean, label=f"{path[11:]}", color='orange')
        #axs[1].set_facecolor('black')
        #axs[1].scatter(phases, los_mean)

        colors1 = get_color_gradient(color1, color2, 20)
        colors2 = get_color_gradient(color2, color1, 20)
        for i in range(0, 19):
            first_range = np.linspace(0, 160, 20)
            second_range = np.linspace(200, 360, 20)
            axs[0].axvspan(first_range[i], first_range[i+1], facecolor=colors1[i], alpha=0.5)
            axs[0].axvspan(second_range[i], second_range[i+1], facecolor=colors2[i], alpha=0.5)
            axs[0].axvspan(160, 200, facecolor='black', alpha=0.5)
            axs[1].axvspan(first_range[i], first_range[i + 1], facecolor=colors1[i], alpha=0.5)
            axs[1].axvspan(second_range[i], second_range[i+1], facecolor=colors2[i], alpha=0.5)
            axs[1].axvspan(160, 200, facecolor='black', alpha=0.5)

    axs[0].vlines([50, 175, 300], ymin=np.min([np.min(x) for x in dens_mean_suparray]),
                  ymax=np.max([np.max(x) for x in dens_mean_suparray]), color='white', linestyles='dashed')
    axs[1].vlines([50, 175, 300], ymin=np.min([np.min(x) for x in los_mean_suparray]),
                  ymax=np.max([np.max(x) for x in los_mean_suparray]), color='white', linestyles='dashed')

    axs[0].set_ylabel(r"$\bar{n}$ [cm$^{-3}$]")
    axs[1].set_ylabel(r"$\bar{N}$ [cm$^{-2}$]")
    axs[1].set_xlabel(r"exomoon phase $\phi \in$ [0, 360]")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', framealpha=1)
    plt.tight_layout()
    plt.show()


#for path in PATHS:
#    #plot_run(path)
#    plot_run(path, top_down=False, LOS=True)

#plot_run()
generate_phase_curves()
#read_pkl()