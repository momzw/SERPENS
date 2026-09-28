"""Physical grain diagnostics for SI SERPENS archives, without survivor renormalization."""

import json
import os
from numbers import Integral

import h5py
import numpy as np
from scipy.spatial import QhullError

from src import DTFE, DTFE3D
from src.grains import GRAIN_METADATA_FIELDS, GrainPopulation, particle_kind, validate_grain_record


QUANTITIES = ('number', 'mass', 'area')
WEIGHT_UNITS = {'number': '1', 'mass': 'kg', 'area': 'm^2'}
REMOVAL_FIELDS = ('removal_cause', 'removal_time', 'removal_interval_start')


def read_grain_archive(filename):
    """Read and validate all birth/loss records, including particles no longer present.

    Missing version/grain groups identify legacy gas archives. Grain-specific
    fields without an explicit kind are errors, never silently treated as gas.
    """
    if not os.path.exists(filename):
        return 0, {}, {}
    records = {}
    populations = {}
    with h5py.File(filename, 'r') as archive:
        version = archive.attrs.get('metadata_version', 0)
        if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral) or version not in (0, 1):
            raise ValueError(f'Unsupported grain metadata version: {version!r}')
        fields = GRAIN_METADATA_FIELDS + REMOVAL_FIELDS
        for field in fields + ('grain_populations',):
            if field in archive and not isinstance(archive[field], h5py.Group):
                raise ValueError(f'Malformed grain metadata group: {field}')
        if 'grain_populations' in archive:
            for key, dataset in archive['grain_populations'].items():
                try:
                    population = GrainPopulation.from_dict(json.loads(dataset[()]))
                    population_id = int(key)
                    if population_id < 0:
                        raise ValueError('negative population ID')
                    populations[population_id] = population.to_dict()
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f'Malformed grain population {key}: {exc}') from exc
        hashes = set()
        for field in fields:
            if field in archive:
                hashes.update(archive[field].keys())
        for key in hashes:
            try:
                record = {field: archive[field][key][()] for field in fields
                          if field in archive and key in archive[field]}
                validate_grain_record(record)
                if particle_kind(record) == 0:
                    continue
                if version != 1:
                    raise ValueError('grain records require metadata_version 1')
                particle_hash = int(key)
                if not 0 <= particle_hash <= np.iinfo(np.uint32).max:
                    raise ValueError('hash must be an unsigned 32-bit integer')
                if record['grain_population_id'] not in populations:
                    raise ValueError('grain_population_id has no grain_populations definition')
                loss_fields = [field in record for field in REMOVAL_FIELDS]
                if any(loss_fields):
                    if not all(loss_fields):
                        raise ValueError('incomplete grain removal record')
                    cause = record['removal_cause']
                    if isinstance(cause, bytes):
                        cause = cause.decode('utf-8')
                    if not isinstance(cause, str) or not cause:
                        raise ValueError('removal_cause must be a nonempty string')
                    record['removal_cause'] = cause
                    for field in REMOVAL_FIELDS[1:]:
                        value = record[field]
                        if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
                            raise ValueError(f'{field} must be finite')
                    if not record['serpens_creation_time'] <= record['removal_interval_start'] <= record['removal_time']:
                        raise ValueError('removal times must follow birth and interval start')
                records[particle_hash] = record
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f'Malformed grain record for hash {key}: {exc}') from exc
    return int(version), populations, records


def grain_selection(metadata, population_id=None, source_hash=None, radius_range=None):
    """Return a hash-aligned mask; radius_range is inclusive grain size in metres."""
    mask = metadata['particle_kind'] == 1
    for field, selection in [('grain_population_id', population_id), ('source_hash', source_hash)]:
        if selection is not None:
            values = np.atleast_1d(selection)
            values = [getattr(value, 'value', value) for value in values]
            mask &= np.isin(metadata[field], values)
    if radius_range is not None:
        bounds = np.asarray(radius_range, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError('radius_range must be finite nonnegative (minimum, maximum) grain radii in metres')
        mask &= (metadata['grain_radius_m'] >= bounds[0]) & (metadata['grain_radius_m'] <= bounds[1])
    return mask


def physical_weights(metadata, quantity):
    if quantity not in QUANTITIES:
        raise ValueError(f'Unknown grain quantity {quantity!r}; choose number, mass, or area')
    weights = np.asarray(metadata['grain_number_weight'], dtype=float).copy()
    if quantity == 'mass':
        weights *= metadata['grain_mass_kg']
    elif quantity == 'area':
        weights *= np.pi * metadata['grain_radius_m'] ** 2
    if not np.all(np.isfinite(weights)):
        raise ValueError(f'Nonfinite physical grain {quantity} weights')
    return weights


def density_units(quantity, dimension):
    if quantity not in QUANTITIES:
        raise ValueError(f'Unknown grain quantity {quantity!r}; choose number, mass, or area')
    if quantity == 'area':
        return '1' if dimension == 2 else 'm^-1'
    return ('kg ' if quantity == 'mass' else '') + f'm^-{dimension}'


def estimate_field(positions, velocities, hashes, metadata, quantity, dimension):
    if isinstance(dimension, (bool, np.bool_)) or dimension not in (2, 3):
        raise ValueError('Grain field dimension must be 2 or 3')
    dimension = int(dimension)
    positions = np.ascontiguousarray(positions[:, :dimension])
    velocities = np.ascontiguousarray(velocities[:, :dimension])
    weights = physical_weights(metadata, quantity)
    result = dict(positions=positions, density=np.empty(0), weights=weights,
                  hashes=hashes.copy(), quantity=quantity, dimension=dimension,
                  units=density_units(quantity, dimension), estimator=None)
    if len(positions) == 0:
        return result
    if len(positions) < dimension + 1:
        raise ValueError(f'Grain DTFE in {dimension}D requires at least {dimension + 1} points; got {len(positions)}')
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError('Grain DTFE requires finite positions and velocities')
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError(f'Grain DTFE has duplicate positions in {dimension}D; no jitter is applied')
    if np.linalg.matrix_rank(positions - positions[0]) < dimension:
        raise ValueError(f'Grain DTFE geometry is degenerate in {dimension}D; no jitter is applied')
    try:
        estimator = (DTFE if dimension == 2 else DTFE3D).DTFE(positions, velocities, weights)
    except (QhullError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        raise ValueError(f'Grain DTFE geometry is degenerate or numerically ill-conditioned in {dimension}D; '
                         'no jitter is applied') from exc
    if not np.all(np.isfinite(estimator.rho)) or not np.all(np.isfinite(estimator.Drho)):
        raise ValueError(f'Grain DTFE geometry is degenerate or produces nonfinite density in {dimension}D')
    result.update(density=estimator.rho.copy(), estimator=estimator)
    return result


def radial_profile(positions, metadata, bins, quantity, center):
    weights = physical_weights(metadata, quantity)
    center = np.zeros(2) if center is None else np.asarray(center, dtype=float)
    if center.shape not in ((2,), (3,)) or not np.all(np.isfinite(center)):
        raise ValueError('center must contain two or three finite SI coordinates')
    center = center[:2].copy()
    radii = np.linalg.norm(positions[:, :2] - center, axis=1)
    if not np.all(np.isfinite(radii)):
        raise ValueError('Grain radial profile requires finite positions')
    if isinstance(bins, Integral) and not isinstance(bins, (bool, np.bool_)):
        if bins <= 0:
            raise ValueError('bins must be positive')
        edges = np.linspace(0., max(float(radii.max()) if len(radii) else 0., 1.), bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) \
                or edges[0] < 0 or np.any(np.diff(edges) <= 0):
            raise ValueError('bins must be strictly increasing finite nonnegative SI radius edges')
    integrated = np.histogram(radii, bins=edges, weights=weights)[0]
    annulus_areas = np.pi * np.diff(edges ** 2)
    return dict(bin_edges=edges, radii=0.5 * (edges[:-1] + edges[1:]), center=center,
                integrated_weights=integrated, column_density=integrated / annulus_areas,
                annulus_areas=annulus_areas, quantity=quantity, units=density_units(quantity, 2),
                weight_units=WEIGHT_UNITS[quantity], radius_units='m')


def historical_budget(records, snapshot_hashes, time, population_id=None, source_hash=None, radius_range=None):
    hashes = np.array(list(records), dtype=np.uint32)
    metadata = {field: np.array([record[field] for record in records.values()]) for field in GRAIN_METADATA_FIELDS}
    selected = grain_selection(metadata, population_id, source_hash, radius_range)
    present = np.isin(hashes, snapshot_hashes)
    created = selected & (present | (metadata['serpens_creation_time'] < time))
    retained = created & present
    removed = np.zeros(len(hashes), dtype=bool)
    causes = {}
    for index, record in enumerate(records.values()):
        if created[index] and not present[index] and 'removal_time' in record and record['removal_time'] <= time:
            removed[index] = True
            causes.setdefault(record['removal_cause'], np.zeros(len(hashes), dtype=bool))[index] = True
    unaccounted = created & ~retained & ~removed
    weights = {quantity: physical_weights(metadata, quantity) for quantity in QUANTITIES}

    def totals(mask):
        return {quantity: float(values[mask].sum()) for quantity, values in weights.items()}

    result = {name: totals(mask) for name, mask in [('injected', created), ('retained', retained),
                                                  ('removed', removed), ('unaccounted', unaccounted)]}
    result['removed_by_cause'] = {cause: totals(mask) for cause, mask in causes.items()}
    result['closure_residual'] = {quantity: result['injected'][quantity] - result['retained'][quantity]
                                - result['removed'][quantity] for quantity in QUANTITIES}
    result['unaccounted_hashes'] = hashes[unaccounted]
    result['units'] = WEIGHT_UNITS.copy()
    result['time'] = float(time)
    return result