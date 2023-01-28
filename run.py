import numpy as np
import shutil
import os
from serpens_analyzer import SerpensAnalyzer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


paths = ['simulation-HD-189733-ExoEarth',
         'simulation-HD-189733-ExoEnce',
         'simulation-HD-189733-ExoIo',
         'simulation-HD-209458-ExoEarth',
         'simulation-HD-209458-ExoEnce',
         'simulation-HD-209458-ExoIo',
         'simulation-WASP-49-ExoEarth',
         'simulation-WASP-49-ExoEnce',
         'simulation-WASP-49-ExoIo',
         'simulation-WASP-39-ExoEarth',
         'simulation-WASP-39-ExoEnce',
         'simulation-WASP-39-ExoIo'
         ]

colors = ['skyblue', 'skyblue', 'skyblue', 'sandybrown', 'sandybrown', 'sandybrown',
          'tomato', 'tomato', 'tomato', 'royalblue', 'royalblue', 'royalblue']

paths = paths[:3]
colors= colors[:3]

means = []
maxes = []
for i, path in enumerate(paths):

    print(f"Started {path[11:]}")

    shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/archive.bin', f'{os.getcwd()}')
    shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/hash_library.pickle', f'{os.getcwd()}')
    shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/Parameters.pickle', f'{os.getcwd()}')
    shutil.copy2(f'{os.getcwd()}/schedule_archive/{path}/Parameters.txt', f'{os.getcwd()}')

    sa = SerpensAnalyzer(save_output=False, folder_name=path[11:])

    _, _, max2d, mean2d = sa.logDensities(np.arange(20, 140, 1))
    means.append(mean2d)
    maxes.append(max2d)

    #sa.top_down(timestep=np.arange(5, 85, 5), d=2, colormesh=True, scatter=False, triplot=True, show=False,
    #            lvlmin=0*np.log(10), lvlmax=18 * np.log(10), lim=6, celest_colors=[colors[i]], trialpha=.5)

    #sa.top_down(timestep=np.arange(100, 280, 50), d=2, colormesh=True, scatter=False, triplot=True, show=False,
    #            lvlmin=0 * np.log(10), lvlmax=18 * np.log(10), lim=6, celest_colors=[colors[i]], trialpha=.5)

    #sa.los(timestep=np.arange(5, 85, 5), show=False, show_planet=True,
    #       lvlmin=0*np.log(10), lvlmax=18 * np.log(10), lim=6, celest_colors=[colors[i]])

    #sa.los(timestep=np.arange(100, 280, 50), show=False, show_planet=True,
    #       lvlmin=0 * np.log(10), lvlmax=18 * np.log(10), lim=6, celest_colors=[colors[i]])

    #sa.los(timestep=np.arange(5, 85, 5), show=False, show_planet=False, show_moon=False,
    #       lvlmin=0*np.log(10), lvlmax=18 * np.log(10), lim=6)

    #sa.los(timestep=np.arange(100, 280, 50), show=False, show_planet=False, show_moon=False,
    #       lvlmin=0 * np.log(10), lvlmax=18 * np.log(10), lim=6)

fig, axs = plt.subplots(2, 1, figsize=(15, 10))
axs[0].plot(np.arange(20, 140, 1), means[0], label=r'$\overline{N}$  Exo-Earth', color='blue')
axs[0].plot(np.arange(20, 140, 1), means[1], label=r'$\overline{N}$  Exo-Enceladus', color='cyan')
axs[0].plot(np.arange(20, 140, 1), means[2], label=r'$\overline{N}$  Exo-Io', color='orange')
axs[0].legend(loc="upper right", fontsize="x-large")
axs[0].grid(True)

axs[1].plot(np.arange(20, 140, 1), maxes[0], label=r'$N_{max}$  Exo-Earth', color='blue')
axs[1].plot(np.arange(20, 140, 1), maxes[1], label=r'$N_{max}$  Exo-Enceladus', color='cyan')
axs[1].plot(np.arange(20, 140, 1), maxes[2], label=r'$N_{max}$  Exo-Io', color='orange')
axs[1].legend(loc="upper right", fontsize="x-large")
axs[1].grid(True)

axs[0].set_ylabel(r"LOS columns log N [cm$^{-2}$]", fontsize='x-large')
axs[1].set_ylabel(r"LOS columns log N [cm$^{-2}$]", fontsize='x-large')
axs[0].set_xlabel("Timestep (40 correspond to one orbit)", fontsize='x-large')
axs[1].set_xlabel("Timestep (40 correspond to one orbit)", fontsize='x-large')

fig.suptitle(r"LOS Column Density Evolution")
fig.tight_layout()

plt.savefig("Test.png")
plt.show()



