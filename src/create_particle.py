import numpy as np
from parameters import Parameters

from scipy.optimize import fmin
from scipy.stats import truncnorm, maxwell, norm, rv_continuous


# ====================================================================================================================================================================

Params = Parameters()
moon_exists = Params.int_spec["moon"]


# ====================================================================================================================================================================



def random_pos(source_r, lat_dist, long_dist, num=1, **kwargs):
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
    latitudes = np.zeros(num)
    longitudes = np.zeros(num)
    if lat_dist in valid_dist:
        if valid_dist[lat_dist] == 0:
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            center = kwargs.get("loc_lat", 0)
            std = kwargs.get("std_lat", 1)
            a, b = (lower - center) / std, (upper - center) / std
            for i in range(num):
                latitudes[i] = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            for i in range(num):
                latitudes[i] = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid latitude distribution encountered in positional calculation")

    if long_dist in valid_dist:
        if valid_dist[long_dist] == 0:
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            center = kwargs.get("loc_long", 0)
            std = kwargs.get("std_long", 1)
            a, b = (lower - center) / std, (upper - center) / std
            for i in range(num):
                longitudes[i] = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            for i in range(num):
                longitudes[i] = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid longitude distribution encountered in positional calculation")

    # Spherical Coordinates. x towards Sun. Change y sign to preserve direction of rotation.
    # Regarding latitude: In spherical coordinates 0 = Northpole, pi = Southpole

    x = source_r * np.cos(longitudes) * np.sin(np.pi / 2 - latitudes)
    y = source_r * np.sin(longitudes) * np.sin(np.pi / 2 - latitudes)
    z = source_r * np.cos(np.pi / 2 - latitudes)

    pos = np.array([x, y, z]).T

    return pos, latitudes, longitudes


def random_temp(source, temp_min, temp_max, latitude, longitude):
    """
    Returns a random temperature depending on implemented model.
    :param temp_min: float. Lowest temperature on the moon
    :param temp_max: float. Highest temperature on the moon
    :param latitude: float
    :param longitude: float
    :return: temp: float
    """

    longitude_wrt_sun = longitude + np.arctan2(source[0][1], source[0][0])

    if not Params.therm_spec["spherical_symm_ejection"]:
        # Coordinate system relevant. If x-axis away from star a longitude -np.pi / 2 < longitude_wrt_sun < np.pi / 2 points away from the star!
        # Need to change temperature-longitude dependence.
        # (refer Wurz, P., 2002, "Monte-Carlo simulation of Mercury's exosphere"; -np.pi / 2 < longitude_wrt_sun < np.pi / 2)
        if np.pi / 2 < longitude_wrt_sun < 3 * np.pi / 2:
            temp = temp_min + (temp_max - temp_min) * (np.abs(np.cos(longitude_wrt_sun)) * np.cos(latitude)) ** (1 / 4)
        else:
            temp = np.repeat(temp_min, len(latitude))
    else:
        temp = np.repeat((temp_max + temp_min) / 2, len(latitude))
    return temp


def random_vel_thermal(species_id, temp):
    """
    Gives a random thermal velocity vector for a sodium atom given a local temperature.
    :param temp: float
    :return: vel_Na: ndarray
    """

    species = Params.get_species(id=species_id)

    k_B = 1.380649e-23
    vel = np.zeros((len(temp), 3))
    for i in range(len(temp)):
        scale = np.sqrt((k_B * temp[i]) / species.m)
        v1 = maxwell.rvs(scale=scale)
        v2 = norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
        v3 = norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
        vel[i] = np.array([v1, v2, v3])

    return vel


def random_vel_sputter(species_id, num=1):
    """
    Gives a random sputter velocity vector for an atom given the at the beginning defined sputtering model.
    :return: vel: ndarray. Randomly generated velocity vector depending on defined model.
    """

    species = Params.get_species(id=species_id)
    sput_model = species.sput_spec["sput_model"]

    # ___________________________________________________

    # MAXWELLIAN MODEL
    def model_maxwell():
        model_maxwell_max = species.sput_spec["model_maxwell_max"]

        scale = model_maxwell_max / np.sqrt(2)

        ran_vel_sputter_maxwell = np.zeros((num, 3))
        for i in range(num):
            maxwell_ran_speed = maxwell.rvs(scale=scale)

            ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)
            ran_elev = np.random.default_rng().uniform(0, np.pi / 2)

            v1 = np.cos(ran_azi) * np.sin(ran_elev)
            v2 = np.sin(ran_azi) * np.sin(ran_elev)
            v3 = np.cos(ran_elev)

            ran_vel_sputter_maxwell[i] = maxwell_ran_speed * np.array(
                [v3, v2, -v1])  # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2.
        return ran_vel_sputter_maxwell

    # MODEL 1
    def model_wurz():

        model_wurz_inc_part_speed = species.sput_spec["model_wurz_inc_part_speed"]
        model_wurz_binding_en = species.sput_spec["model_wurz_binding_en"]
        model_wurz_inc_mass_in_amu = species.sput_spec["model_wurz_inc_mass_in_amu"]
        model_wurz_ejected_mass_in_amu = species.sput_spec["model_wurz_ejected_mass_in_amu"]
        u = 1.660539e-27
        m = model_wurz_ejected_mass_in_amu * u

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

        ran_vel_sputter_wurz = np.zero((num,3))
        for i in range(num):
            ran_energy = energy_dist.rvs(E_i, E_b)

            ran_speed = np.sqrt(2 * ran_energy / m)
            ran_elev = elev_dist.rvs()
            ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)

            v1 = np.cos(ran_azi) * np.sin(ran_elev)
            v2 = np.sin(ran_azi) * np.sin(ran_elev)
            v3 = np.cos(ran_elev)

            ran_vel_sputter_wurz[i] = ran_speed * np.array([v3, v2, -v1]) # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2.
        return ran_vel_sputter_wurz

    # MODEL 2
    def model_smyth():

        v_b = species.sput_spec["model_smyth_v_b"]
        v_M = species.sput_spec["model_smyth_v_M"]
        a = species.sput_spec["model_smyth_a"]

        def rejection_method():

            def phi_neg(x):

                def phi(x):
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
                f_pdf = normalization * phi(x)
                return -f_pdf

            vmax = fmin(phi_neg, v_b, full_output=True, disp=False)
            height = -vmax[1]

            v_rv = np.zeros((num,3))
            for i in range(num):
                while True:
                    x_rv = np.random.uniform(0, np.sqrt(v_M**2 - v_b**2))
                    y_rv = np.random.uniform(0, height)
                    if y_rv < -phi_neg(x_rv):
                        break

                ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)
                ran_elev = np.random.default_rng().uniform(0, np.pi / 2)

                v1 = np.cos(ran_azi) * np.sin(ran_elev)
                v2 = np.sin(ran_azi) * np.sin(ran_elev)
                v3 = np.cos(ran_elev)

                v_rv[i] = x_rv * np.array([v3, v2, -v1])  # Rotated, s.t. reference direction along x-axis. Otherwise, the azimuth may point into the source. For same reason ele only goes to pi/2. Hemisphere pointing up -> Hemisphere pointing right
            return v_rv

        ran_vel_sputter_smyth = rejection_method()
        return ran_vel_sputter_smyth

    # ___________________________________________________

    valid_model = {"maxwell": 0, "wurz": 1, "smyth": 2}
    if sput_model in valid_model:
        if valid_model[sput_model] == 0:
            vel = model_maxwell()
        elif valid_model[sput_model] == 1:
            vel = model_wurz()
        else:
            vel = model_smyth()
    else:
        raise ValueError("Invalid sputtering model")

    return vel


def create_particle(species_id, process, source, source_r, num=1, **kwargs):
    """
    Generates a REBOUND particle with random velocity at random position from process given by function argument.
    See the "random_pos" and "random_vel_..." functions for more info on the random position and velocity generation.
    :param process: str. Valid are "thermal" and "sputter".
    :param kwargs: kwargs. Parameters forwarded to random generation functions: temp_midnight, temp_noon, E_incoming, E_bind
    :return: p: rebound particle object.
    """

    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    sim = rebound.Simulation("archive.bin")

    valid_process = {"thermal": 0, "sputter": 1}
    if process not in valid_process:
        raise ValueError("Invalid escaping mechanism encountered in particle creation")

    temp_min = kwargs.get("temp_min", Params.therm_spec['source_temp_min'])
    temp_max = kwargs.get("temp_max", Params.therm_spec['source_temp_max'])

    out = np.zeros((num, 6), dtype="float64")

    if valid_process[process] == 0:
        ran_pos, ran_lat, ran_long = random_pos(source_r, lat_dist="truncnorm", long_dist="uniform", a_long=0,
                                                b_long=2 * np.pi, num = num)
        ran_temp = random_temp(source, temp_min, temp_max, ran_lat, ran_long)

        ran_vel_not_rotated_in_place = random_vel_thermal(species_id, ran_temp)

    else:

        #if moon_exists:
        #    angle_correction = np.arctan2((sim.particles["moon"].y - planet.y),
        #                                  (sim.particles["moon"].x - planet.x))
        #else:
        #    angle_correction = np.arctan2(sim.particles["planet"].y,sim.particles["planet"].x)
        #
        #ran_pos, ran_lat, ran_long = random_pos(sim, lat_dist="uniform", long_dist="truncnorm", a_lat=-np.pi / 2,
        #                                        b_lat=np.pi / 2, a_long=-np.pi / 2 + angle_correction,
        #                                        b_long=3 * np.pi / 2 + angle_correction,
        #                                        loc_long=np.pi / 2 + angle_correction)
        #ran_temp = random_temp(sim, temp_min, temp_max, ran_lat, ran_long)

        ran_pos, ran_lat, ran_long = random_pos(source_r, lat_dist="uniform", long_dist="uniform", num=num)
        ran_vel_not_rotated_in_place = random_vel_sputter(species_id, num=num)

    for part in range(num):
        # Rotation matrix in order to get velocity vector aligned with surface-normal.
        # Counterclockwise along z-axis (local longitude). Clockwise along y-axis (local latitude).
        rot_z = np.array([[np.cos(ran_long[part]), -np.sin(ran_long[part]), 0], [np.sin(ran_long[part]), np.cos(ran_long[part]), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(ran_lat[part]), 0, -np.sin(ran_lat)[part]], [0, 1, 0], [np.sin(ran_lat[part]), 0, np.cos(ran_lat[part])]])
        rot = rot_y @ rot_z

        ran_vel = rot @ ran_vel_not_rotated_in_place[part]

        out[part][:3] = np.array([ran_pos[part][0] + source[0][0], ran_pos[part][1] + source[0][1], ran_pos[part][2] + source[0][2]])
        out[part][3:] = np.array([ran_vel[0] + source[1][0], ran_vel[1] + source[1][1], ran_vel[2] + source[1][2]])

    return out