import numpy as np
from scipy.optimize import fmin
from scipy.stats import truncnorm, maxwell, norm

from src.grains import GrainPopulation, _array, _real
from src.parameters import GLOBAL_PARAMETERS


def generate_grains(population, source_state, source_r, stellar_mass_kg, interval_s, rng):
    """Return inertial SI states (N, 6) and per-tracer grain properties.

    source_state is shaped (2, 3): source position followed by velocity.
    Launch sites are uniform on the sphere at source_r + launch_altitude_m;
    directions are uniform in solid angle on each outward hemisphere.
    Every tracer has equal number weight rate * interval / sum(grain masses),
    so the represented mass is independent of count and injection cadence.
    Zero production or interval returns empty arrays without consuming rng.
    """
    if not isinstance(population, GrainPopulation):
        raise TypeError("population must be a GrainPopulation")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    interval_s = _real(interval_s, "interval_s", minimum=0)
    source_r = _real(source_r, "source_r", positive=True)
    stellar_mass_kg = _real(stellar_mass_kg, "stellar_mass_kg", positive=True)
    source_state = _array(source_state, "source_state")
    if source_state.shape != (2, 3):
        raise ValueError("source_state must have shape (2, 3)")
    launch_radius = _real(source_r + population.launch_altitude_m, "launch radius", positive=True)
    if population.dust_mass_per_sec == 0 or interval_s == 0:
        properties = population.properties(np.empty(0), stellar_mass_kg)
        properties["grain_number_weight"] = np.empty(0)
        return np.empty((0, 6)), properties

    represented_mass = _real(population.dust_mass_per_sec * interval_s,
                             "represented mass", positive=True)
    radii = population.sample_radii(rng)
    properties = population.properties(radii, stellar_mass_kg)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        sampled_mass = _real(np.sum(properties["grain_mass_kg"]), "sampled mass", positive=True)
        weight = _real(represented_mass / sampled_mass, "grain number weight", positive=True)
    properties["grain_number_weight"] = np.full(len(radii), weight)
    speeds = population.launch_speeds(radii)

    def uniform_sphere():
        z = rng.uniform(-1, 1, len(radii))
        azimuth = rng.uniform(0, 2 * np.pi, len(radii))
        equatorial = np.sqrt(np.maximum(0, 1 - z ** 2))
        return np.column_stack((equatorial * np.cos(azimuth), equatorial * np.sin(azimuth), z))

    normals = uniform_sphere()
    directions = uniform_sphere()
    # Folding an independent uniform sphere gives uniform outward solid angle.
    inward = np.einsum("ij,ij->i", normals, directions) < 0
    directions[inward] *= -1
    with np.errstate(over="ignore", invalid="ignore"):
        positions = source_state[0] + launch_radius * normals
        velocities = source_state[1] + speeds[:, None] * directions
    states = _array(np.hstack((positions, velocities)), "grain launch states")
    return states, properties


def random_pos(radius, lat_dist='uniform', long_dist='uniform', n_samples=1, **kwargs):
    """
    Keyword Arguments
    -----------------
    Distribution shape parameters:
    a_lat, b_lat, loc_lat, std_lat
    a_long, b_long, loc_long, std_long
    """
    # Coordinates:
    # Inertial system Cartesian coordinates. x-axis points from star away, y in direction of orbit.
    valid_dist = {"truncnorm": 0, "uniform": 1}
    assert lat_dist in valid_dist
    assert long_dist in valid_dist

    if valid_dist[lat_dist] == 0:   # truncnorm
        lower = kwargs.get("lowest_latitude", -np.pi / 2)
        upper = kwargs.get("highest_latitude", np.pi / 2)
        center = kwargs.get("center_latitude", 0)
        std = kwargs.get("std_latitude", 1)
        a, b = (lower - center) / std, (upper - center) / std
        latitudes = truncnorm.rvs(a, b, loc=center, scale=std, size=n_samples)
    elif valid_dist[lat_dist] == 1:     # uniform
        lower = kwargs.get("lowest_latitude", -np.pi / 2)
        upper = kwargs.get("highest_latitude", np.pi / 2)
        latitudes = np.random.uniform(lower, upper, size=n_samples)
    else:
        print("An unexpected sampling error occurred")
        latitudes = None

    if valid_dist[long_dist] == 0:  # truncnorm
        lower = kwargs.get("lowest_longitude", -np.pi)
        upper = kwargs.get("highest_longitude", np.pi)
        center = kwargs.get("center_longitude", 0)
        std = kwargs.get("std_longitude", 1)
        a, b = (lower - center) / std, (upper - center) / std
        longitudes = truncnorm.rvs(a, b, loc=center, scale=std, size=n_samples)
    elif valid_dist[long_dist] == 1:    # uniform
        lower = kwargs.get("lowest_longitude", -np.pi)
        upper = kwargs.get("highest_longitude", np.pi)
        longitudes = np.random.uniform(lower, upper, size=n_samples)
    else:
        print("An unexpected sampling error occurred")
        longitudes = None

    # Spherical Coordinates. x towards Sun. Change y sign to preserve direction of rotation.
    # Regarding latitude: In spherical coordinates 0 = Northpole, pi = Southpole
    x = radius * np.cos(longitudes) * np.sin(np.pi / 2 - latitudes)
    y = radius * np.sin(longitudes) * np.sin(np.pi / 2 - latitudes)
    z = radius * np.cos(np.pi / 2 - latitudes)

    return np.column_stack((x, y, z)), latitudes, longitudes


def random_temperature(source, temp_min, temp_max, latitude, longitude):
    """
    Calculate temperature for a sampled position on the source.
    Temperature will be the mean value of minimal and maximal temperature if spherical symmetric ejection is assumed,
    otherwise a model by Wurz, P., 2002, "Monte-Carlo simulation of Mercury's exosphere", will be used.

    Arguments
    ---------
    source : array-like with shape (6,)
        State vector of the source.#
    temp_min : float
        Minimum temperature on the source.
    temp_max : float
        Maximum temperature on the source.
    latitude : float
        Latitude at which to calculate temperature.
    longitude : float
        Longitude at which to calculate temperature
    """
    longitude_wrt_sun = longitude - np.arctan2(source[0][1], source[0][0])
    if not GLOBAL_PARAMETERS.get("spherical_symm_ejection", True):
        # Coordinate system relevant. If x-axis away from star a longitude -np.pi / 2 < longitude_wrt_sun < np.pi / 2 points away from the star!
        # (refer Wurz, P., 2002, "Monte-Carlo simulation of Mercury's exosphere"; -np.pi / 2 < longitude_wrt_sun < np.pi / 2)
        if np.pi / 2 < longitude_wrt_sun < 3 * np.pi / 2:
            temp = temp_min + (temp_max - temp_min) * (np.abs(np.cos(longitude_wrt_sun)) * np.cos(latitude)) ** (1 / 4)
        else:
            temp = np.repeat(temp_min, len(latitude))
    else:
        temp = np.repeat((temp_max + temp_min) / 2, len(latitude))
    return temp


def random_thermal_velocity(species_id, temp):
    """
    Generates a random thermal velocity vector

    Arguments
    ---------
    species_id : int
        id of the species for which to sample the velocity. Relevant for the Maxwell distribution.
    temp : float
        Local temperature.
    """
    #Params = Parameters()
    #species = Params.get_species(id=species_id)
    species = [s for s in GLOBAL_PARAMETERS.get('species') if s.id == species_id][0]

    k_B = 1.380649e-23
    #vel = np.zeros((len(temp), 3))
    #for i in range(len(temp)):
    #    scale = np.sqrt((k_B * temp[i]) / species.m)
    #    v1 = maxwell.rvs(scale=scale)
        #v2 = scale * norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
        #v3 = scale * norm.rvs(scale=1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
        #vel[i] = np.array([v1, v2, v3])

    v = maxwell.rvs(
        scale=np.sqrt((k_B * temp) / species.m)
    )
    # Generate directions
    ran_azi = np.random.uniform(0, 2 * np.pi, len(temp))
    ran_elev = np.random.uniform(0, np.pi / 2, len(temp))

    v1 = np.cos(ran_azi) * np.sin(ran_elev)
    v2 = np.sin(ran_azi) * np.sin(ran_elev)
    v3 = np.cos(ran_elev)

    # Rotate output, s.t. reference direction along x-axis. Otherwise, azimuth may point into source. For same reason
    # elevation goes to pi/2. Hemisphere pointing up -> hemisphere pointing right.
    velocities = np.stack([v3, v2, -v1], axis=-1) * np.array(v)[:, None]

    return velocities


def random_sputter_velocity(species_id, n_samples=1):
    #Params = Parameters()
    #species = Params.get_species(id=species_id)
    species = [s for s in GLOBAL_PARAMETERS.get('species') if s.id == species_id][0]

    v_b = species.sput_spec["v_b"]
    v_M = species.sput_spec["v_M"]
    a = species.sput_spec["a"]

    def phi(x):
        f_v = 1 / v_b * (x / v_b) ** 3 * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** a * (
                    1 - np.sqrt((x ** 2 + v_b ** 2) / (v_M ** 2)))
        return f_v

    def phi_int(x):
        bracket = (1 + x ** 2) ** (5 / 2) * v_b / v_M / (2 * a - 5) - (1 + x ** 2) ** (3 / 2) * v_b / v_M / (
                    2 * a - 3) - x ** 4 / (2 * a - 4) - a * x ** 2 / (2 * (a - 2) * (a - 1)) - 1 / (
                              2 * (a - 2) * (a - 1))
        f_v_int = bracket * (1 + x ** 2) ** (-a)
        return f_v_int

    upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
    normalization = 1 / (phi_int(upper_bound) - phi_int(0))
    phi_normalized = lambda x: normalization * phi(x)

    # Compute height
    vmax = fmin(lambda x: -phi_normalized(x), v_b, full_output=True, disp=False)
    height = -vmax[1]

    # Generate samples
    x_rv = []
    while len(x_rv) < n_samples:
        x_candidates = np.random.uniform(0, np.sqrt(v_M ** 2 - v_b ** 2), n_samples)
        y_candidates = np.random.uniform(0, height, n_samples)
        accepted = x_candidates[y_candidates < phi_normalized(x_candidates)]
        x_rv.extend(accepted.tolist())

    x_rv = x_rv[:n_samples]  # Limit to requested number of samples

    # Generate directions
    ran_azi = np.random.uniform(0, 2 * np.pi, n_samples)
    ran_elev = np.random.uniform(0, np.pi / 2, n_samples)

    v1 = np.cos(ran_azi) * np.sin(ran_elev)
    v2 = np.sin(ran_azi) * np.sin(ran_elev)
    v3 = np.cos(ran_elev)

    # Rotate output, s.t. reference direction along x-axis. Otherwise, azimuth may point into source. For same reason
    # elevation goes to pi/2. Hemisphere pointing up -> hemisphere pointing right.
    velocities = np.stack([v3, v2, -v1], axis=-1) * np.array(x_rv)[:, None]

    # ___________________________________________________

    return velocities


def generate_particles(species_id, process, source, source_r, n_samples=1, **kwargs):
    """
    Generate a set of state vectors containing position and velocity for new particles.
    The velocity of the particle depends on the physical process that generates it.

    Arguments
    ---------
    species_id : int
        id of the species the particles belong to.
    process : str
        Physical process responsible for particle creation (valid are "thermal" and "sputter").
    source : array-like (shape (6,))
        State vector of the source object.
    source_r : float
        Radius of the source object.
    n_samples : int
        Number of state vectors (particles) to generate.

    Keyword Arguments
    -----------------
    temp_min : float
        Minimum temperature on the source (midnight)
    temp_max : float
        Maximum temperature on the source (noon)
    """

    valid_process = {"thermal": 0, "sputter": 1}
    assert process in valid_process, "Invalid escaping mechanism encountered in particle creation"

    temp_min = kwargs.get("temp_min", GLOBAL_PARAMETERS.get('source_temp_min', 0))
    temp_max = kwargs.get("temp_max", GLOBAL_PARAMETERS.get('source_temp_max', 0))

    # Thermal
    if valid_process[process] == 0:
        positions, latitudes, longitudes = random_pos(
            1.1 * source_r, lat_dist="uniform", long_dist="uniform",
            n_samples=n_samples
        )
        ran_temp = random_temperature(source, temp_min, temp_max, latitudes, longitudes)
        velocities_not_rotated = random_thermal_velocity(species_id, ran_temp)
    # Sputter
    else:
        positions, latitudes, longitudes = random_pos(
            1.1 * source_r, lat_dist="uniform", long_dist="uniform",
            n_samples=n_samples
        )
        velocities_not_rotated = random_sputter_velocity(species_id, n_samples=n_samples)

    cos_latitudes = np.cos(latitudes)
    sin_latitudes = np.sin(latitudes)
    cos_longitudes = np.cos(longitudes)
    sin_longitudes = np.sin(longitudes)

    rot_y = np.array([
        [cos_latitudes, np.zeros(n_samples), -sin_latitudes],  # Rows of the rotation matrix
        [np.zeros(n_samples), np.ones(n_samples), np.zeros(n_samples)],
        [sin_latitudes, np.zeros(n_samples), cos_latitudes]
    ]).transpose(2, 0, 1)  # Transpose to (n_samples, 3, 3)

    # Rotate counterclockwise around z-axis (longitude rotation)
    rot_z = np.array([
        [cos_longitudes, -sin_longitudes, np.zeros(n_samples)],  # Rows of the rotation matrix
        [sin_longitudes, cos_longitudes, np.zeros(n_samples)],
        [np.zeros(n_samples), np.zeros(n_samples), np.ones(n_samples)]
    ]).transpose(2, 0, 1)  # Transpose to (n_samples, 3, 3)

    # Combine rotations: rot_y @ rot_z, batched matrix multiplication
    rot_matrices = np.einsum('nij,njk->nik', rot_y, rot_z)  # Batched matrix

    # Rotate velocities
    velocities = np.einsum('nij,nj->ni', rot_matrices, velocities_not_rotated)

    # Combine into the output array
    state_vectors = np.hstack((positions, velocities))

    return state_vectors