import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

def calculate_vb(temperatures, v_M):

    def func(x, v_M, T):
        v_m = np.sqrt(3 * 1.380649e-23 * T / (39 * 1.660539066e-27))
        v_m *= 1e-3
        xi = 1/(v_M/np.sqrt(v_m**2 + x**2) - 1)
        xi_term = np.sqrt(14/9 + 1/3 * xi - 1)
        return v_m - x/xi_term

    if isinstance(temperatures, (list, np.ndarray)) and isinstance(v_M, (list, np.ndarray)):
        assert len(temperatures) == len(v_M)
        roots = []
        for i in range(len(temperatures)):
            root = sp.fsolve(func, 2, args=(v_M[i], temperatures[i]))
            roots.append(root)
        return roots
    elif isinstance(temperatures, (int, float)) and isinstance(v_M, (int, float)):
        root = sp.fsolve(func, 2, args=(v_M, temperatures))
        return root
    else:
        print("Invalid types for temperatures and/or maximal velocities")
        return
#calculate_vb(temperatures=[7754, 1400, 1120, 1740, 963, 1200, 1450, 1322], v_M=[46.62, 15.24, 11.86, 16.02, 12.43, 23.8, 17.82, 13.35])


def calculate_dsync():
    import objects
    w_init = 1.77e-4
    alpha = 0.26
    Q_diss = 1e9
    gc = 6.6743e-11
    tau_sync = 100e6 * 3.156e7
    d_sync = []
    tau = []
    for i in range(2, 11):
        celest = objects.celestial_objects(moon=True, set=i)
        star_mass = celest["star"]["m"]
        planet_mass = celest["planet"]["m"]
        planet_radius = celest["planet"]["r"]
        planet_semimajor = celest["planet"]["a"]
        d = (9/4 * 1/(alpha*Q_diss) * gc*planet_mass/planet_radius**3 * 1/w_init * star_mass**2/planet_mass**2 * planet_radius**6 * tau_sync)**(1/6)
        tau_at_orbit = 4/9 * alpha * Q_diss * planet_radius**3/(gc*planet_mass) * w_init * planet_mass**2/star_mass**2 * (planet_semimajor/planet_radius)**6
        tau.append(tau_at_orbit * 3.17098e-8 * 1e-6)
        d_sync.append(d/planet_semimajor)
    return d_sync, tau


def calculate_alfven(eta=.3):
    import objects
    gc = 6.6743e-11
    ages = np.array([5, 4.3, 3.5, 3.6, 5, 3, 2, 9.4, 6.3])
    spt = np.array(['G', 'K', 'G', 'G', 'G', 'F', 'K', 'G', 'K'])
    lxuv = np.zeros(9)
    for i, type in enumerate(spt):
        if type == 'G':
            lxuv[i] = 0.19 * 10**29.35 * ages[i]**(-1.69)
        elif type == 'K':
            lxuv[i] = 0.234 * 10**28.87 * ages[i]**(-1.72)
        elif type == 'F':
            lxuv[i] = 0.155 * 10**29.83 * ages[i]**(-1.72)

    dmdt_therm = []
    r_alf = []
    for j in range(2, 11):
        celest = objects.celestial_objects(moon=True, set=j)
        star_mass = celest["star"]["m"]
        planet_mass = celest["planet"]["m"]
        planet_radius = celest["planet"]["r"]
        semimajor = celest["planet"]["a"]
        xi = semimajor * (1/(3*planet_radius**3) * planet_mass/star_mass)**(1/3)
        k = 1 - 3/2 * xi - 1/(2 * xi**3)
        mass_loss = eta/(4 * gc * k) * planet_radius**3/(semimajor**2 * planet_mass) * lxuv[j-2]*1e-7 * 1e3
        dmdt_therm.append(mass_loss)
        r_alf.append((1e6/-mass_loss)**(1/5) * 19.8 * 69911e3 / planet_radius)

    return np.asarray(dmdt_therm), np.asarray(r_alf)


def calculate_params(celestial_system, stellar_temperature=None, lineofsight_observed_cm2=None, lifetime=None,
                     exosphere_temperature=None, stellar_age_gyr=None, stellar_spectral_type=None):
    import objects

    gc = 6.6743e-11
    sb_constant = 5.67051e-8
    amu = 1.660539066e-27
    w_init = 1.77e-4
    alpha = 0.26
    Q_diss = 1e9
    gc = 6.6743e-11
    tau_sync = 100e6 * 3.156e7

    celest = objects.celestial_objects(moon=True, set=celestial_system)
    star_mass = celest["star"]["m"]
    star_radius = celest["star"]["r"]
    planet_mass = celest["planet"]["m"]
    planet_radius = celest["planet"]["r"]
    planet_semimajor = celest["planet"]["a"]
    moon_semimajor = celest["moon"]["a"]

    # Hill radius
    r_hill = planet_semimajor * (planet_mass/(3*star_mass))**(1/3)

    # Irradiance for calculation of beta
    beta = None
    if stellar_temperature is not None:
        irradiance = (star_radius/planet_semimajor)**2 * sb_constant * stellar_temperature**4
        beta = 0.269 * irradiance * 1e-5 + 0.745  # linear interpolation on account of HD189 and HD209

    # Tidal synchronization calculations
    d = (9 / 4 * 1 / (
                alpha * Q_diss) * gc * planet_mass / planet_radius ** 3 * 1 / w_init * star_mass ** 2 / planet_mass ** 2 * planet_radius ** 6 * tau_sync) ** (
                    1 / 6)
    sync_timescale_at_orbit = 4 / 9 * alpha * Q_diss * planet_radius ** 3 / (
                gc * planet_mass) * w_init * planet_mass ** 2 / star_mass ** 2 * (planet_semimajor / planet_radius) ** 6
    sync_timescale = sync_timescale_at_orbit * 3.17098e-8 * 1e-6
    d_sync = d / planet_semimajor

    # Thermal exoplanet mass-loss for Alfvén radius calculation
    alfven_radius = None
    log_planet_mass_loss_therm = None
    if stellar_spectral_type is not None and stellar_age_gyr is not None:
        if stellar_spectral_type == 'G':
            lxuv = 0.19 * 10 ** 29.35 * stellar_age_gyr ** (-1.69)
        elif stellar_spectral_type == 'K':
            lxuv = 0.234 * 10 ** 28.87 * stellar_age_gyr ** (-1.72)
        elif stellar_spectral_type == 'F':
            lxuv = 0.155 * 10 ** 29.83 * stellar_age_gyr ** (-1.72)
        else:
            print("Invalid spectral type.")
            lxuv = None
        heating_efficiency = 0.3
        xi = planet_semimajor * (1 / (3 * planet_radius ** 3) * planet_mass / star_mass) ** (1 / 3)
        k = 1 - 3 / 2 * xi - 1 / (2 * xi ** 3)
        planet_mass_loss_therm = heating_efficiency / (4 * gc * k) * planet_radius ** 3 / (planet_semimajor ** 2 * planet_mass) * lxuv * 1e-7 * 1e3
        alfven_radius = (1e6 / -planet_mass_loss_therm) ** (1 / 5) * 19.8 * 69911e3 / planet_radius
        log_planet_mass_loss_therm = np.log10(-planet_mass_loss_therm)

    # Mass-loss rate calculation
    mass_loss = None
    if lineofsight_observed_cm2 is not None:
        mass_loss = np.log10(lineofsight_observed_cm2 * 10000 * np.pi * star_radius ** 2 * 39 * amu / lifetime)

    # Calculation of sputtering parameters
    vb = None
    vM = None
    if exosphere_temperature is not None:
        orbital_period = 2 * np.pi * np.sqrt(planet_semimajor ** 3 / (gc * star_mass))
        corotating_velocity = 2 * np.pi / orbital_period * moon_semimajor
        satellite_velocity = np.sqrt(gc * planet_mass / moon_semimajor)
        vM = np.abs(satellite_velocity - corotating_velocity)
        vb = calculate_vb(exosphere_temperature, vM)

    print(f"Hill radius: {r_hill/planet_radius} [R_p]")
    print(f"Alfvén radius: {alfven_radius} [R_p]")
    print(f"Tidal synchronization timescale: {sync_timescale} [Myr]")
    print(f"Tidal synchronization distance for timescale = 100 Myr: {d_sync} [a_p]")
    print(f"Exoplanet thermal mass-loss: {log_planet_mass_loss_therm} [kg/s]")
    print(f"Mass-loss rate: {mass_loss} [log kg/s]")
    print(f"Radiation beta: {beta}")
    print(f"Sputtering parameter v_b: {vb[0]} [km/s]")
    print(f"Sputtering parameter v_M: {vM/1000} [km/s]")


def velocity_distributions():
    from scipy.stats import maxwell

    temp_eq = 1400  # W49
    temp_tidal = 1902   # W49
    m = 23 * 1.660539066e-27    # Na
    k_B = 1.38066e-23

    # MAXWELL:
    maxwell_scale_eq = np.sqrt((k_B * temp_eq) / m)
    maxwell_scale_tidal = np.sqrt((k_B * temp_tidal) / m)
    x = np.linspace(0, 8000, 300)

    # SPUTTERING:

    def phi(x, scale):
        v_M = 15.24*1000
        v_b = calculate_vb(temp_eq, v_M/1000) * 1000
        a = scale

        def phi_unnorm(x):
            f_v = 1 / v_b * (x / v_b) ** 3 \
                  * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** a \
                  * (1 - np.sqrt((x ** 2 + v_b ** 2) / (v_M ** 2)))
            return f_v

        def phi_int(x):
            integral_bracket = (1 + x ** 2) ** (5 / 2) * v_b / v_M / (2 * a - 5) - (1 + x ** 2) ** (
                    3 / 2) * v_b / v_M / (2 * a - 3) - x ** 4 / (2 * (a - 2)) - a * x ** 2 / (
                                       2 * (a - 2) * (a - 1)) - 1 / (2 * (a - 2) * (a - 1))

            f_v_integrated = integral_bracket * (1 + x ** 2) ** (-a)
            return f_v_integrated

        upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
        normalization = 1 / (phi_int(upper_bound) - phi_int(0))
        f_pdf = normalization * phi_unnorm(x)
        return f_pdf

    fig, ax = plt.subplots(1, 1, figsize=(12,6), dpi=150)
    #fig.suptitle("Velocity Distributions", fontsize='x-large')
    ax.plot(x, maxwell.pdf(x, scale=maxwell_scale_eq), label=r'Maxwell $T_{\mathrm{eq}}$', color='blue')
    ax.plot(x, maxwell.pdf(x, scale=maxwell_scale_tidal), label=r'Maxwell $T_{\mathrm{tidal}}$', color='cornflowerblue')
    ax.plot(x, phi(x, scale=3), label=r'Sputtering $T_{\mathrm{eq}}$, $\alpha=3$', color='orange')
    ax.plot(x, phi(x, scale=7/3), label=r'Sputtering $T_{\mathrm{eq}}$, $\alpha=7/3$', color='red')

    from matplotlib import ticker
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_ylabel(r"Probability density [10$^{-4}$]", fontsize=22)
    ax.set_xlabel("Velocity in m/s", fontsize=22)
    ax.set_ylim(-0.5e-4, 8.7e-4)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(prop={'size': 22}, framealpha=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig("veldist.png")
    plt.show()

    #import plotly.express as px
    #df = pd.DataFrame({
    #    'x': x,
    #    'Maxwell T_eq': maxwell.pdf(x, scale=maxwell_scale_eq),
    #    'Maxwell T_tidal': maxwell.pdf(x, scale=maxwell_scale_tidal),
    #    'Sputtering alpha 3': phi(x, scale=3),
    #    'Sputtering alpha 7/3': phi(x, scale=7/3)
    #})
    #
    #fig = px.line(df, x='x', y=df.columns[1:])
    #fig.write_image("fig1.png")
    #fig.show()