import rebound
import numpy as np
from init import Parameters

# ====================================================================================================================================================================

Params = Parameters()
therm_Params = Params.therm_spec
sput_Params = Params.sput_spec
moon_exists = Params.int_spec["moon"]

# Thermal evaporation parameters
# ---------------------
source_temp_max = therm_Params["source_temp_max"]
source_temp_min = therm_Params["source_temp_min"]
spherical_symm_ejection = therm_Params["spherical_symm_ejection"]

# Sputtering model
# ---------------------
sput_model = sput_Params["sput_model"]  # Valid inputs: maxwell, wurz, smyth.

# Sputtering model shape parameters
# ---------------------
model_maxwell_max = sput_Params["model_maxwell_max"]

model_wurz_inc_part_speed = sput_Params["model_wurz_inc_part_speed"]
model_wurz_binding_en = sput_Params["model_wurz_binding_en"]  # See table 1, in: Kudriavtsev Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".
model_wurz_inc_mass_in_amu = sput_Params["model_wurz_inc_mass_in_amu"]
model_wurz_ejected_mass_in_amu = sput_Params["model_wurz_ejected_mass_in_amu"]

model_smyth_v_b = sput_Params["model_smyth_v_b"]       # "low cutoff" speed to prevent the slowest nonescaping atoms from dominating the distribution (see Wilson et al. 2002)
model_smyth_v_M = sput_Params["model_smyth_v_M"]     # Maximum velocity achievable. Proportional to plasma velocity (see Wilson et al. 2002)
model_smyth_a = sput_Params["model_smyth_a"]       # Speed distribution shape parameter

# ====================================================================================================================================================================



def random_pos(sim, lat_dist, long_dist, **kwargs):
    """
    This function allows for different distributions for latitude and longitude according to which positions on the moon are randomly generated.
    :param lat_dist: str. Valid are "truncnorm" and "uniform".
    :param long_dist: str. Valid are "truncnorm" and "uniform".
    :param kwargs: kwargs. Distribution shape parameters: a_lat, b_lat, loc_lat, std_lat, a_long, b_long, loc_long, std_long
    :return: pos: ndarray, latitude: float, longitude: float

    """
    # Coordinates:
    # Inertial system Cartesian coordinates. x-axis points from star away, y in direction of orbit.
    valid_dist = {"truncnorm": 0, "uniform": 1}
    if lat_dist in valid_dist:
        if valid_dist[lat_dist] == 0:
            from scipy.stats import truncnorm
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            center = kwargs.get("loc_lat", 0)
            std = kwargs.get("std_lat", 1)
            a, b = (lower - center) / std, (upper - center) / std
            latitude = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            latitude = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid latitude distribution encountered in positional calculation: " % lat_dist)

    if long_dist in valid_dist:
        if valid_dist[long_dist] == 0:
            from scipy.stats import truncnorm
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            center = kwargs.get("loc_long", 0)
            std = kwargs.get("std_long", 1)
            a, b = (lower - center) / std, (upper - center) / std
            longitude = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            longitude = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid longitude distribution encountered in positional calculation: " % long_dist)

    # Spherical Coordinates. x towards Sun. Change y sign to preserve direction of rotation.
    # Regarding latitude: In spherical coordinates 0 = Northpole, pi = Southpole
    if moon_exists:
        x = sim.particles["moon"].r * np.cos(longitude) * np.sin(np.pi / 2 - latitude)
        y = sim.particles["moon"].r * np.sin(longitude) * np.sin(np.pi / 2 - latitude)
        z = sim.particles["moon"].r * np.cos(np.pi / 2 - latitude)

    else:
        x = sim.particles["planet"].r * np.cos(longitude) * np.sin(np.pi / 2 - latitude)
        y = sim.particles["planet"].r * np.sin(longitude) * np.sin(np.pi / 2 - latitude)
        z = sim.particles["planet"].r * np.cos(np.pi / 2 - latitude)

    pos = np.array([x, y, z])

    return pos, latitude, longitude


def random_temp(sim, temp_min, temp_max, latitude, longitude):
    """
    Returns a random temperature depending on implemented model.
    :param temp_min: float. Lowest temperature on the moon
    :param temp_max: float. Highest temperature on the moon
    :param latitude: float
    :param longitude: float
    :return: temp: float
    """
    if moon_exists:
        longitude_wrt_sun = longitude + np.arctan2(sim.particles["moon"].y, sim.particles["moon"].x)
    else:
        longitude_wrt_sun = longitude + np.arctan2(sim.particles["planet"].y, sim.particles["planet"].x)

    if not spherical_symm_ejection:
        # Coordinate system relevant. If x-axis away from star a longitude -np.pi / 2 < longitude_wrt_sun < np.pi / 2 points away from the star!
        # Need to change temperature-longitude dependence.
        # (refer Wurz, P., 2002, "Monte-Carlo simulation of Mercury's exosphere"; -np.pi / 2 < longitude_wrt_sun < np.pi / 2)
        if np.pi / 2 < longitude_wrt_sun < 3 * np.pi / 2:
            temp = temp_min + (temp_max - temp_min) * (np.abs(np.cos(longitude_wrt_sun)) * np.cos(latitude)) ** (
                    1 / 4)
        else:
            temp = temp_min
    else:
        temp = (temp_max + temp_min) / 2
    return temp


def random_vel_thermal(species, temp):
    """
    Gives a random thermal velocity vector for a sodium atom given a local temperature.
    :param temp: float
    :return: vel_Na: ndarray
    """
    from scipy.stats import maxwell, norm
    v1 = maxwell.rvs()
    v2 = norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
    v3 = norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)

    k_B = 1.380649e-23
    vel_Na = np.sqrt((k_B * temp) / species.m) * np.array([v1, v2, v3])

    return vel_Na


def random_vel_sputter():
    """
    Gives a random sputter velocity vector for an atom given the at the beginning defined sputtering model.
    :return: vel: ndarray. Randomly generated velocity vector depending on defined model.
    """
    from scipy.stats import rv_continuous

    # MAXWELLIAN MODEL
    def model_maxwell():
        from scipy.stats import maxwell

        scale = model_maxwell_max / np.sqrt(2)

        maxwell_ran_speed = maxwell.rvs(scale=scale)

        ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)
        ran_elev = np.random.default_rng().uniform(0, np.pi / 2)

        v1 = np.cos(ran_azi) * np.sin(ran_elev)
        v2 = np.sin(ran_azi) * np.sin(ran_elev)
        v3 = np.cos(ran_elev)

        ran_vel_sputter_maxwell = maxwell_ran_speed * np.array(
            [v3, v2, -v1])  # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2.
        return ran_vel_sputter_maxwell

    # MODEL 1
    def model_wurz():
        class _sputter_gen(rv_continuous):
            def _pdf(self, x, E_inc, E_bin):
                normalization = (E_inc + E_bin) ** 2 / E_inc ** 2
                f_E = normalization * 2 * E_bin * x / ((x + E_bin) ** 3)
                return f_E

        class _elevation_gen(rv_continuous):
            def _pdf(self, x):
                normalization = 4 / np.pi  # Inverse of integral of cos^2 from 0 to pi/2
                f_alpha = normalization * np.cos(x) ** 2
                return f_alpha

        E_i = 1/2 * model_wurz_inc_mass_in_amu * model_wurz_inc_part_speed**2
        E_b = model_wurz_binding_en

        elev_dist = _elevation_gen(a=0, b=np.pi / 2)
        energy_dist = _sputter_gen(a=0, b=E_i, shapes='E_inc, E_bin')

        ran_energy = energy_dist.rvs(E_i, E_b)
        u = 1.660539e-27
        m = model_wurz_ejected_mass_in_amu * u
        ran_speed = np.sqrt(2 * ran_energy / m)

        ran_elev = elev_dist.rvs()
        ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)

        v1 = np.cos(ran_azi) * np.sin(ran_elev)
        v2 = np.sin(ran_azi) * np.sin(ran_elev)
        v3 = np.cos(ran_elev)

        ran_vel_sputter_model1 = ran_speed * np.array([v3, v2, -v1]) # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2.
        return ran_vel_sputter_model1

    # MODEL 2
    def model_smyth():
        class sputter_gen2(rv_continuous):
            def _pdf(self, x, alpha, v_b, v_M):
                def model2func(x, alpha, v_b, v_M):
                    f_v = 1 / v_b * (x / v_b) ** 3 \
                          * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** alpha \
                          * (1 - np.sqrt((x ** 2 + v_b ** 2) / (v_M ** 2)))
                    return f_v

                def model2func_int(x, alpha, v_b, v_M):

                    integral_bracket = (1 + x**2)**(5/2) * v_b/v_M / (2*alpha-5) - (1 + x**2)**(3/2) * v_b/v_M / (2*alpha-3) - x**4 / (2*(alpha-2)) - alpha*x**2 / (2*(alpha-2)*(alpha-1)) - 1 / (2*(alpha-2)*(alpha-1))

                    f_v_integrated = integral_bracket * (1+x**2)**(-alpha)

                    return f_v_integrated

                #normalization = 1 / (model2func_int(v_M, a, v_b, v_M) - model2func_int(v_b, a, v_b, v_M))
                upper_bound = np.sqrt((v_M/v_b)**2 - 1)
                normalization = 1 / (model2func_int(upper_bound, a, v_b, v_M) - model2func_int(0, a, v_b, v_M))
                f_pdf = normalization * model2func(x, a, v_b, v_M)
                return f_pdf

        v_b = model_smyth_v_b
        v_M = model_smyth_v_M
        a = model_smyth_a
        model2_vel_dist = sputter_gen2(a=0.0, b=v_M)
        model2_ran_speed = model2_vel_dist.rvs(alpha=a, v_b=v_b, v_M=v_M)

        ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)
        ran_elev = np.random.default_rng().uniform(0, np.pi/2)

        v1 = np.cos(ran_azi) * np.sin(ran_elev)
        v2 = np.sin(ran_azi) * np.sin(ran_elev)
        v3 = np.cos(ran_elev)

        ran_vel_sputter_model2 = model2_ran_speed * np.array(
            [v3, v2, -v1])  # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2.
        return ran_vel_sputter_model2

    # ___________________________________________________

    valid_model = {"maxwell": 0, "wurz": 1, "smyth": 2}
    if sput_model in valid_model:
        if valid_model[sput_model] == 0:
            vel = model_maxwell()
        elif valid_model[sput_model] == 1:
            vel = model_wurz()
        else:
            vel = model_smyth()

            """
            TESTING
            
            import matplotlib.pyplot as plt
            import matplotlib
            from scipy.integrate import quad
            matplotlib.use('TkAgg')

            vels = []
            x_vals = np.linspace(0, model_smyth_v_M, 3000)
            for i in range(3000):
                vels.append(np.linalg.norm(model_smyth()))

            a = Params.sput_spec['model_smyth_a']
            v_M = Params.sput_spec['model_smyth_v_M']
            v_b = Params.sput_spec['model_smyth_v_b']
            def model2func(x, a, v_b, v_M):
                f_v = 1 / v_b * (x / v_b) ** 3 \
                      * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** a \
                      * (1 - np.sqrt((x ** 2 + v_b ** 2) / (v_M ** 2)))
                return f_v
            def model2func_int(x, a, v_b, v_M):
                integral_bracket = (1 + x ** 2) ** (5 / 2) * v_b / v_M / (2 * a - 5) - (1 + x ** 2) ** (
                        3 / 2) * v_b / v_M / (
                                           2 * a - 3) - x ** 4 / (2 * (a - 2)) - a * x ** 2 / (
                                           2 * (a - 2) * (a - 1)) - 1 / (
                                           2 * (a - 2) * (a - 1))

                f_v_integrated = integral_bracket * (1 + x ** 2) ** (-a)
                return f_v_integrated
            upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
            normalization = 1 / (model2func_int(upper_bound, a, v_b, v_M) - model2func_int(0, a, v_b, v_M))
            f_pdf = normalization * model2func(x_vals, a, v_b, v_M)

            plt.figure()
            plt.hist(vels, density=True, bins=100)
            plt.plot(x_vals, f_pdf, c='r')
            plt.show()
            breakpoint = True
            """

    else:
        raise ValueError("Invalid sputtering model: " % sput_model)

    return vel


def create_particle(species, process, **kwargs):
    """
    Generates a REBOUND particle with random velocity at random position from process given by function argument.
    See the "random_pos" and "random_vel_..." functions for more info on the random position and velocity generation.
    :param process: str. Valid are "thermal" and "sputter".
    :param kwargs: kwargs. Parameters forwarded to random generation functions: temp_midnight, temp_noon, E_incoming, E_bind
    :return: p: rebound particle object.
    """
    sim = rebound.Simulation("archive.bin")

    valid_process = {"thermal": 0, "sputter": 1}
    if process in valid_process:
        temp_min = kwargs.get("temp_min", source_temp_min)
        temp_max = kwargs.get("temp_max", source_temp_max)
        if valid_process[process] == 0:
            ran_pos, ran_lat, ran_long = random_pos(sim, lat_dist="truncnorm", long_dist="uniform", a_long=0,
                                                    b_long=2 * np.pi)
            ran_temp = random_temp(sim, temp_min, temp_max, ran_lat, ran_long)

            ran_vel_not_rotated_in_place = random_vel_thermal(species, ran_temp)

        else:
            if moon_exists:
                angle_correction = np.arctan2((sim.particles["moon"].y - sim.particles["planet"].y),
                                              (sim.particles["moon"].x - sim.particles["planet"].x))
            else:
                angle_correction = np.arctan2(sim.particles["planet"].y,sim.particles["planet"].x)

            #ran_pos, ran_lat, ran_long = random_pos(sim, lat_dist="uniform", long_dist="truncnorm", a_lat=-np.pi / 2,
            #                                        b_lat=np.pi / 2, a_long=-np.pi / 2 + angle_correction,
            #                                        b_long=3 * np.pi / 2 + angle_correction,
            #                                        loc_long=np.pi / 2 + angle_correction)
            ran_pos, ran_lat, ran_long = random_pos(sim, lat_dist="uniform", long_dist="uniform")
            ran_temp = random_temp(sim, temp_min, temp_max, ran_lat, ran_long)

            ran_vel_not_rotated_in_place = random_vel_sputter()
    else:
        raise ValueError("Invalid escaping mechanism encountered in particle creation: " % process)

    # Rotation matrix in order to get velocity vector aligned with surface-normal.
    # Counterclockwise along z-axis (local longitude). Clockwise along y-axis (local latitude).
    rot_z = np.array([[np.cos(ran_long), -np.sin(ran_long), 0], [np.sin(ran_long), np.cos(ran_long), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(ran_lat), 0, -np.sin(ran_lat)], [0, 1, 0], [np.sin(ran_lat), 0, np.cos(ran_lat)]])
    rot = rot_y @ rot_z

    ran_vel = rot @ ran_vel_not_rotated_in_place

    # source position and velocity:
    if moon_exists:
        source_x, source_y, source_z = sim.particles["moon"].x, sim.particles["moon"].y, sim.particles["moon"].z
        source_vx, source_vy, source_vz = sim.particles["moon"].vx, sim.particles["moon"].vy, sim.particles["moon"].vz
    else:
        source_x, source_y, source_z = sim.particles["planet"].x, sim.particles["planet"].y, sim.particles["planet"].z
        source_vx, source_vy, source_vz = sim.particles["planet"].vx, sim.particles["planet"].vy, sim.particles["planet"].vz

    p = rebound.Particle()
    p.x, p.y, p.z = ran_pos[0] + source_x, ran_pos[1] + source_y, ran_pos[2] + source_z
    p.vx, p.vy, p.vz = ran_vel[0] + source_vx, ran_vel[1] + source_vy, ran_vel[2] + source_vz

    return p