import numpy as np
import rebound
import os
from tqdm import tqdm
import pandas as pd

from serpens_analyzer import SerpensAnalyzer


def orbit_sol():

    G = 6.6743e-11

    # Set a range of semi-major axes to check (and pericenters for when e!=0):
    a = 1.01 * 69911e3 * np.linspace(1.2, 1.8, 200000)
    w = np.linspace(0, 2*np.pi, 10000)

    # Orbital periods
    T = 2 * np.pi * np.sqrt(a**3/(G*0.365*1.898e27))

    # Fixed phases after t0
    phases = np.array([0.65, 0.4])

    # Set dates to check:
    dates = np.array([["2015-12-07 04:31", 2457363.69313672],
                      ["2016-01-01 05:22", 2457388.72868279],
                      ["2019-11-11 13:32", 2458799.06791922],
                      ["2022-09-25 09:00", 2459847.79251231],
                      ["2022-10-20 07:30", 2459872.81516997],
                      ["2022-11-03 05:00", 2459886.71188757],
                      ["2022-11-14 08:00", 2459897.83748205],
                      ["2022-11-17 03:00", 2459900.62927906],
                      ["2022-11-28 06:00", 2459911.75471299],
                      ["2022-12-12 04:00", 2459925.67171136],
                      ["2022-12-23 07:04", 2459936.79957287],
                      ["2022-12-26 02:00", 2459939.58845666],
                      ["2016-01-15 03:10", 2457402.63672867],
                      ["2023-01-03 10:08", 2459947.92726799],
                      ["2023-01-06 04:53", 2459950.70847165],
                      ["2023-01-08 23:39", 2459953.49035973],
                      ["2023-01-11 18:25", 2459956.27223777],
                      ["2023-01-14 13:11", 2459959.05410612],
                      ["2023-01-17 07:56", 2459961.83527034],
                      ["2023-01-20 02:42", 2459964.61711949],
                      ["2023-01-22 21:28", 2459967.39895940],
                      ["2023-01-25 16:13", 2459970.18009599],
                      ["2023-01-28 10:59", 2459972.96191840],
                      ["2023-01-31 05:44", 2459975.74303804],
                      ["2023-02-03 00:30", 2459978.52484407],
                      ["2023-02-05 19:16", 2459981.30664256],
                      ["2023-02-08 14:02", 2459984.08843367],
                      ["2023-02-11 08:48", 2459986.87021793],
                      ["2023-02-14 03:33", 2459989.65130116],
                      ["2023-02-16 22:19", 2459992.43307256],
                      ["2023-02-19 17:05", 2459995.21483827],
                      ["2023-02-22 11:50", 2459997.99590420],
                      ["2023-02-25 06:36", 2460000.77765964],
                      ["2023-02-28 01:22", 2460003.55941073],
                      ])

    # BJD and time after first BJD (t0)
    t = dates[:,1].astype(float)
    dt = np.asarray([(dates[:,1].astype(float)[i] - dates[:,1].astype(float)[0]) * 86400 for i in range(np.size(t))])
    sol_a = []
    sol_T = []
    sol_w = []
    coll_phases = []

    e = 0
    for i in tqdm(range(np.size(a)), "Checking a and w"):
        # Calculate phase angles for the fixed phases with phi(t0)=0.67
        m1 = ((2*np.pi/T[i] * dt[1] + 0.67 * 2 * np.pi) % (2*np.pi)) / (2*np.pi)    # Angle modulo 2pi mapped to [0,1]
        m2 = ((2*np.pi/T[i] * dt[2] + 0.67 * 2 * np.pi) % (2*np.pi)) / (2*np.pi)

        if not e == 0:
            nu1 = rebound.M_to_f(e, m1) / (2*np.pi)
            nu2 = rebound.M_to_f(e, m2) / (2*np.pi)

            for j in range(np.size(w)):
                nu1 = nu1 + w[j]
                nu2 = nu2 + w[j]

                if np.abs((nu1 - phases[0])) < 0.005:
                    if np.abs((nu2 - phases[1])) < 0.005:
                        sol_a.append(a[i] / (1.01 * 69911e3))
                        sol_T.append(T[i])
                        sol_w.append(w[j])
                        all_phases = [0.67]
                        all_phases.append(nu1)
                        all_phases.append(nu2)
                        for ti in dt[3:]:
                            all_phases.append(((2 * np.pi / T[i] * ti + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                        coll_phases.append(all_phases)
        else:
            if np.abs((m1 - phases[0])) < 0.005:
                if np.abs((m2 - phases[1])) < 0.005:
                    sol_a.append(a[i] / (1.01 * 69911e3))
                    sol_T.append(T[i])
                    sol_w.append(0.)

                    # Calculate/append all phases for the semi-major axis solution found.
                    all_phases = [0.67]
                    all_phases.append(m1)
                    all_phases.append(m2)
                    for ti in dt[3:]:
                        all_phases.append(((2 * np.pi / T[i] * ti + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                    coll_phases.append(all_phases)  # Collect phases for file output

    # Saving data...
    pericenters = np.asarray([f"w = {np.round(sol_w[j],3)}" for j in range(np.size(sol_w))])
    header = ["date", "BJD", "delta_t [s]"]
    for j in range(np.size(sol_a)):
        header.append(f"phase for a = {np.round(sol_a[j],3)}")

    phase_outputs = np.vstack((header[3:], pericenters, np.round(np.asarray(coll_phases).T,5)))
    all_output = np.hstack((np.vstack((header[:3], np.array(["-", "-", "-"]), np.vstack((dates[:,0], t, dt)).T)), phase_outputs))
    np.savetxt("temporary.txt", all_output, fmt="%-20s", delimiter=" ", newline='\n')

    df = pd.DataFrame(all_output)
    df.to_excel('temporary.xlsx', index=False, header=False)


def pngToGif(path, fps):
    """
    TOOL TO COMBINE PNG TO GIF
    """
    import imageio

    files = []
    for file in os.listdir(path):
        if file.startswith('ColumnDensity_TopDown_'):
            cutoff = file[::-1].find('_')
            files.append(file[:-cutoff])

    date = path[path.find('/')+1:path.find('_')]
    writer = imageio.get_writer(f'sim_{date}.mp4', fps=fps, macro_block_size=None)

    for im in sorted(files, key = len):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) and f'{im}' in file:
                im2 = os.path.join(path, file)
                writer.append_data(imageio.v3.imread(im2))
    writer.close()


sa = SerpensAnalyzer(save_output=True)
sa.top_down(timestep=280, d=2, colormesh=False, scatter=True, triplot=False, show=False)
sa.top_down(timestep=280, d=3, colormesh=False, scatter=True, triplot=False, show=False)
sa.top_down(timestep=280, d=2, colormesh=True, scatter=False, triplot=True, show=False)
sa.top_down(timestep=280, d=3, colormesh=True, scatter=False, triplot=True, show=False)
