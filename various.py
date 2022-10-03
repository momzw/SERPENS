import numpy as np
import rebound
from tqdm import tqdm
import pandas as pd
import openpyxl



# TESTING
G = 6.6743e-11
a = 1.01 * 69911e3 * np.linspace(1.2, 1.8, 20000)
w = np.linspace(0, 2*np.pi, 10000)
T = 2 * np.pi * np.sqrt(a**3/(G*0.365*1.898e27))
phases = np.array([0.65, 0.4])
dates = np.array(["2015-12-07 04:31", "2016-01-01 05:22", "2019-11-11 13:32", "2022-12-23 07:04", "2023-01-31 05:44", "2023-02-03 00:30", "2023-02-14 03:32", "2020-12-16 03:14", "2016-01-15 03:10"])
t = np.array([2457363.69313672, 2457388.72868279, 2458799.06791922, 2459936.79957287, 2459975.74303804, 2459978.52484407, 2459989.65060670, 2459199.6398463, 2457402.63672867])
dt = np.array([0, 2163071.18045, 124016381.208, 222316396.083, 225681111.474, 225921459.515, 226882725.406, 158625795.708, 3364726.34448])
sol_a = []
sol_T = []
sol_w = []
coll_phases = []
for i in tqdm(range(np.size(a)), "Checking a and w"):
    m1 = ((2*np.pi/T[i] * dt[1] + 0.67 * 2 * np.pi) % (2*np.pi))
    m2 = ((2*np.pi/T[i] * dt[2] + 0.67 * 2 * np.pi) % (2*np.pi))
    nu1 = rebound.M_to_f(0.004, m1) / (2*np.pi)
    nu2 = rebound.M_to_f(0.004, m2) / (2*np.pi)

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
                all_phases.append(((2 * np.pi / T[i] * dt[3] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                all_phases.append(((2 * np.pi / T[i] * dt[4] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                all_phases.append(((2 * np.pi / T[i] * dt[5] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                all_phases.append(((2 * np.pi / T[i] * dt[6] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                all_phases.append(((2 * np.pi / T[i] * dt[7] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                all_phases.append(((2 * np.pi / T[i] * dt[8] + 0.67 * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi))
                coll_phases.append(all_phases)

pericenters = np.asarray([f"w = {np.round(sol_w[j],3)}" for j in range(np.size(sol_w))])
header = ["date", "BJD", "delta_t [s]"]
for j in range(np.size(sol_a)):
    header.append(f"phase for a = {np.round(sol_a[j],3)}")

phase_outputs = np.vstack((header[3:], pericenters, np.round(np.asarray(coll_phases).T,5)))
all_output = np.hstack((np.vstack((header[:3], np.array(["-", "-", "-"]), np.vstack((dates, t, dt)).T)), phase_outputs))
np.savetxt("temporary.txt", all_output, fmt="%-20s", delimiter=" ", newline='\n')

df = pd.DataFrame(all_output)
df.to_excel('temporary.xlsx', index=False, header=False)

