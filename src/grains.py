"""SI grain properties and number-distribution sampling, independent of the simulator.

No material, stellar luminosity, or radiation efficiency is calibrated here.
Grains are homogeneous spheres; their charge uses the isolated-sphere capacitance.

Omitted physics:
* grain-grain collisions and fragmentation
* coagulation
* sublimation and sputtering erosion
* time-dependent charging
* plasma and gas drag
* radiative transfer and shadowing
* wavelength-dependent optical properties
* dust back-reaction and self-gravity

"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np


G_SI = 6.67430e-11
C_M_S = 299792458.0
EPSILON_0_F_M = 8.8541878128e-12

GRAIN_METADATA_FIELDS = (
    "particle_kind", "grain_population_id", "grain_radius_m", "grain_mass_kg",
    "grain_number_weight", "beta", "q_over_m", "source_hash",
    "serpens_creation_time",
)


def _real(value, name, minimum=None, positive=False):
    """Type and value validation for real numbers"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _integer(value, name, minimum=0):
    """Type and value validation for integers"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _array(value, name, positive=False, nonnegative=False):
    """Type and value validation for arrays"""
    if not isinstance(value, np.ndarray):
        if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
            raise ValueError(f"{name} must not contain booleans")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def grain_mass_kg(radius_m, density_kg_m3):
    """Return spherical grain mass in kg: (4 pi / 3) rho a**3."""
    radius_m = _array(radius_m, "radius_m", positive=True)
    density_kg_m3 = _real(density_kg_m3, "density_kg_m3", positive=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = (4 * np.pi / 3) * density_kg_m3 * radius_m ** 3
    return _array(result, "grain_mass_kg", positive=True)


def radiation_pressure_beta(radius_m, density_kg_m3, stellar_mass_kg, luminosity_w, q_pr):
    """Return dimensionless radiation/gravity ratio 3 L Qpr / (16 pi c G M rho a)."""
    radius_m = _array(radius_m, "radius_m", positive=True)
    density_kg_m3 = _real(density_kg_m3, "density_kg_m3", positive=True)
    stellar_mass_kg = _real(stellar_mass_kg, "stellar_mass_kg", positive=True)
    luminosity_w = _real(luminosity_w, "luminosity_w", minimum=0)
    q_pr = _array(q_pr, "q_pr", nonnegative=True)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = (3 / (16 * np.pi * C_M_S * G_SI)) * (luminosity_w / stellar_mass_kg) \
            * q_pr / density_kg_m3 / radius_m
    return _array(result, "beta", nonnegative=True)


def grain_q_over_m(radius_m, density_kg_m3, potential_v=0):
    """Return signed charge/mass in C/kg: 3 epsilon_0 V / (rho a**2)."""
    radius_m = _array(radius_m, "radius_m", positive=True)
    density_kg_m3 = _real(density_kg_m3, "density_kg_m3", positive=True)
    potential_v = _real(potential_v, "potential_v")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = (3 * EPSILON_0_F_M * potential_v / density_kg_m3) / radius_m / radius_m
    return _array(result, "q_over_m")


@dataclass(frozen=True) # Avoid accidental modification, and to guarantee a consistent value
class GrainPopulation:
    """Spherical-grain population.

    Radii follow the bounded *number* distribution dN/da proportional to
    a**(-size_slope), not a mass-weighted distribution. Equal bounds are
    monodisperse. Speeds are speed_ref_m_s * (a/radius_ref_m)**(-speed_exponent),
    optionally clipped to the supplied speed bounds (a positive exponent
    makes larger grains slower).

    q_pr is a nonnegative constant or an increasing sequence of (radius_m,
    q_pr) pairs. Tables must cover the population's full radius interval;
    interpolation is linear in radius and extrapolation is rejected.
    radiation_source is an explicit, nonblank body name, resolved by the caller.
    """

    name: str
    density_kg_m3: float
    radius_min_m: float
    radius_max_m: float
    size_slope: float
    tracers_per_injection: int
    dust_mass_per_sec: float
    speed_ref_m_s: float
    radius_ref_m: float
    speed_exponent: float
    launch_altitude_m: float
    luminosity_w: float
    q_pr: float | tuple[tuple[float, float], ...]
    radiation_source: str = "star"
    potential_v: float = 0.0
    speed_min_m_s: float | None = None
    speed_max_m_s: float | None = None

    def __post_init__(self):
        """
        Setting variables with some guards.
        Using post-init to avoid duplicate code and misaligning defaults.
        """
        for name in ("name", "radiation_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be an explicit nonblank string")

        # Could make the following a loop, but keeping it like this for now.
        # If in loop, use e.g.: object.__setattr__(self, name, _real(getattr(self, name), name, **kwargs))

        # Only positive guard
        object.__setattr__(self, "density_kg_m3", _real(self.density_kg_m3, "density_kg_m3", positive=True))
        object.__setattr__(self, "radius_min_m", _real(self.radius_min_m, "radius_min_m", positive=True))
        object.__setattr__(self, "radius_max_m", _real(self.radius_max_m, "radius_max_m", positive=True))
        object.__setattr__(self, "radius_ref_m", _real(self.radius_ref_m, "radius_ref_m", positive=True))

        # Minimum 0 guard
        object.__setattr__(self, "dust_mass_per_sec", _real(self.dust_mass_per_sec, "dust_mass_per_sec", minimum=0))
        object.__setattr__(self, "speed_ref_m_s", _real(self.speed_ref_m_s, "speed_ref_m_s", minimum=0))
        object.__setattr__(self, "launch_altitude_m", _real(self.launch_altitude_m, "launch_altitude_m", minimum=0))
        object.__setattr__(self, "luminosity_w", _real(self.luminosity_w, "luminosity_w", minimum=0))

        # Minimum 1 guard
        object.__setattr__(
            self, "tracers_per_injection",
            _integer(self.tracers_per_injection, "tracers_per_injection", minimum=1)
        )

        # No additional guard.
        object.__setattr__(self, "size_slope", _real(self.size_slope, "size_slope"))
        object.__setattr__(self, "speed_exponent", _real(self.speed_exponent, "speed_exponent"))
        object.__setattr__(self, "potential_v", _real(self.potential_v, "potential_v"))

        # Handle min-max radii
        if self.radius_min_m > self.radius_max_m:
            raise ValueError("radius_min_m must not exceed radius_max_m")
        for name in ("speed_min_m_s", "speed_max_m_s"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _real(getattr(self, name), name, minimum=0))

        # Handle min-max speeds
        if self.speed_min_m_s is not None and self.speed_max_m_s is not None \
                and self.speed_min_m_s > self.speed_max_m_s:
            raise ValueError("speed_min_m_s must not exceed speed_max_m_s")

        # Handle radiation pressure coefficients
        if isinstance(self.q_pr, Real):
            object.__setattr__(self, "q_pr", _real(self.q_pr, "q_pr", minimum=0))
        else:
            table = _array(self.q_pr, "q_pr table", nonnegative=True)
            if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
                raise ValueError("q_pr table must contain at least two (radius_m, q_pr) pairs")
            if np.any(table[:, 0] <= 0) or np.any(np.diff(table[:, 0]) <= 0):
                raise ValueError("q_pr table radii must be positive and strictly increasing")
            if table[0, 0] > self.radius_min_m or table[-1, 0] < self.radius_max_m:
                raise ValueError("q_pr table must cover the population radius interval")
            object.__setattr__(self, "q_pr", tuple(tuple(float(v) for v in row) for row in table))

    def sample_radii(self, rng=None, count=None):
        """
        Sample radii in metres
        """
        if not isinstance(rng, np.random.Generator):
            # Default rng
            print("Warning, `rng` is not of type np.random.Generator \nUsing default with seed 42.")
            rng = np.random.default_rng(42)

        count = self.tracers_per_injection if count is None else _integer(count, "count")

        if self.radius_min_m == self.radius_max_m:
            return np.full(count, self.radius_min_m)

        u = rng.random(count)
        log_min = np.log(self.radius_min_m)
        log_max = np.log(self.radius_max_m)
        span = log_max - log_min
        power = 1 - self.size_slope
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled_span = power * span
            if power == 0 or abs(scaled_span) < 1e-8:
                # The limit is log-uniform; the first correction retains near-p=1 accuracy.
                log_radius = log_min + span * (u + 0.5 * scaled_span * u * (1 - u))
            elif abs(scaled_span) < 50:
                log_radius = log_min + np.log1p(u * np.expm1(scaled_span)) / power
            elif power > 0:
                log_radius = log_max + np.logaddexp(np.log(u), np.log1p(-u) - scaled_span) / power
            else:
                log_radius = log_min + np.logaddexp(np.log1p(-u), np.log(u) + scaled_span) / power
        radii = np.exp(log_radius)
        radii[u == 0] = self.radius_min_m
        return np.clip(radii, self.radius_min_m, self.radius_max_m)

    def radiation_efficiency(self, radii):
        """Evaluate Qpr without extrapolating a bounded table."""
        radii = _array(radii, "radii", positive=True)
        if isinstance(self.q_pr, float):
            return np.full_like(radii, self.q_pr)
        table = np.asarray(self.q_pr)
        if np.any(radii < table[0, 0]) or np.any(radii > table[-1, 0]):
            raise ValueError("radii lie outside the q_pr table")
        return np.interp(radii, table[:, 0], table[:, 1])

    def properties(self, radii, stellar_mass_kg):
        """Return shape-preserving SI arrays (zero-dimensional for scalar input)."""
        radii = _array(radii, "radii", positive=True)
        return {
            "grain_radius_m": radii.copy(),
            "grain_mass_kg": grain_mass_kg(radii, self.density_kg_m3),
            "beta": radiation_pressure_beta(radii, self.density_kg_m3, stellar_mass_kg,
                                             self.luminosity_w, self.radiation_efficiency(radii)),
            "q_over_m": grain_q_over_m(radii, self.density_kg_m3, self.potential_v),
        }

    def launch_speeds(self, radii):
        """Return launch speed magnitudes in m/s, with optional clipping."""
        radii = _array(radii, "radii", positive=True)
        if self.speed_ref_m_s == 0:
            speeds = np.zeros_like(radii)
        else:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                speeds = np.exp(np.log(self.speed_ref_m_s) - self.speed_exponent
                                * (np.log(radii) - np.log(self.radius_ref_m)))
        if self.speed_min_m_s is not None:
            speeds = np.maximum(speeds, self.speed_min_m_s)
        if self.speed_max_m_s is not None:
            speeds = np.minimum(speeds, self.speed_max_m_s)
        return _array(speeds, "launch speeds", nonnegative=True)

    def to_dict(self):
        """Return an independent JSON-compatible mapping in SI units."""
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        if isinstance(self.q_pr, tuple):
            result["q_pr"] = [list(row) for row in self.q_pr]
        return result

    @classmethod
    def from_dict(cls, record):
        """Deserialize with the same strict validation as direct construction."""
        if not isinstance(record, Mapping):
            raise ValueError("population record must be a mapping")
        return cls(**dict(record))


def particle_kind(record):
    """Return 0 for gas (including legacy records), 1 for explicitly tagged grains.

    Grain-specific fields without a kind are malformed, not legacy gas. Shared
    fields such as beta, q_over_m, source_hash, and creation time may occur in gas.
    """
    if not isinstance(record, Mapping):
        raise ValueError("particle record must be a mapping")
    grain_fields = GRAIN_METADATA_FIELDS[1:5]
    if "particle_kind" not in record:
        if any(name in record for name in grain_fields):
            raise ValueError("grain metadata requires an explicit particle_kind")
        return 0
    kind = _integer(record["particle_kind"], "particle_kind")
    if kind not in (0, 1):
        raise ValueError("particle_kind must be 0 (gas) or 1 (grains)")
    if kind == 0 and any(name in record for name in grain_fields):
        raise ValueError("gas records must not contain grain-specific metadata")
    return kind


def validate_grain_record(record):
    """Validate a grain record; accept legacy/explicit gas records unchanged.

    Population IDs and source hashes are nonnegative integers. Number weights
    and beta are nonnegative; potential and hence q_over_m may have either sign.
    Successful validation returns None; malformed records raise ValueError.
    """
    if particle_kind(record) == 0:
        return
    missing = [name for name in GRAIN_METADATA_FIELDS if name not in record]
    if missing:
        raise ValueError(f"grain record is missing fields: {', '.join(missing)}")
    _integer(record["grain_population_id"], "grain_population_id")
    _integer(record["source_hash"], "source_hash")
    for name in ("grain_radius_m", "grain_mass_kg"):
        _real(record[name], name, positive=True)
    for name in ("grain_number_weight", "beta"):
        _real(record[name], name, minimum=0)
    _real(record["q_over_m"], "q_over_m")
    _real(record["serpens_creation_time"], "serpens_creation_time")