import numpy as np
import rebound
from tqdm import tqdm
import pandas as pd
from serpens_analyzer import SerpensAnalyzer, PhaseCurve
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.style.use('seaborn-dark')


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
                      ["2023-10-11 09:04", 2460228.87978633],
                      ["2023-10-25 06:52", 2460242.78908222],
                      ["2023-11-08 04:41", 2460256.69894613],
                      ["2023-11-19 07:43", 2460267.82587894],
                      ["2023-11-22 02:29", 2460270.60793891],
                      ["2023-12-03 05:32", 2460281.73539485],
                      ["2023-12-14 08:35", 2460292.86269663],
                      ["2023-12-17 03:20", 2460295.64397583],
                      ["2023-12-28 06:23", 2460306.77107229],
                      ["2023-12-31 01:09", 2460309.5529942],
                      ["2024-01-11 04:12", 2460320.67988477],
                      ["2024-01-22 07:15", 2460331.80661763],
                      ["2024-01-25 02:00", 2460334.58775701],
                      ["2024-02-05 05:03", 2460345.71431384],
                      ["2024-02-19 02:52", 2460359.62251892],
                      ["2024-03-04 00:40", 2460373.52989522],
                      ["2024-03-15 03:43", 2460384.65615342],
                      ["2024-03-29 01:31", 2460398.56343303],
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

#sa = SerpensAnalyzer(save_output=False, reference_system="geocentric")

#sa.top_down(timestep=1, d=2, colormesh=False, scatter=True, triplot=False, show=True, smoothing=.5, trialpha=1, lim=5,
#            celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
#            colormap=plt.cm.get_cmap("afmhot"), show_moon=True, lvlmin=-10, lvlmax=15)

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
