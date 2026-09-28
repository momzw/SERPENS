"""Scientific preparation, independent of rendering and display thresholds."""

import copy
import hashlib
import os
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.parameters import GLOBAL_PARAMETERS


@dataclass(frozen=True)
class PlotRequest:
    timestep: int
    view: str = 'planar'
    species_ids: Optional[Tuple[int, ...]] = None
    dimension: int = 3
    source_hash: Optional[str] = None
    max_distance_rp: float = 6.
    kind: str = 'gas'
    quantity: str = 'number'
    population_ids: Optional[Tuple[int, ...]] = None
    radius_range: Optional[Tuple[float, float]] = None
    phase_file: Optional[str] = None
    orbits: float = 1.


@dataclass
class DisplayOptions:
    thresholds: dict = field(default_factory=dict)
    colorscale: str = 'Viridis'
    separate: bool = True
    active_field: Optional[str] = None
    extent: Optional[float] = None
    scatter: bool = True
    mesh: bool = False
    mesh_opacity: float = 0.3
    bodies: bool = True
    orbits: bool = True
    shadow: bool = False
    connection: bool = False
    show_star: bool = False
    log_scale: bool = False
    statistic: str = 'max'
    column_density: bool = True
    particle_density: bool = True


@dataclass
class BodyGeometry:
    name: str
    position: np.ndarray
    radius: float
    primary: bool = False
    star: bool = False
    orbit: Optional[np.ndarray] = None


@dataclass
class FieldData:
    key: str
    name: str
    positions: np.ndarray
    density: np.ndarray
    valid: np.ndarray
    visible: np.ndarray
    simplices: np.ndarray
    units: str
    quantity: str = 'number'
    message: str = ''


@dataclass
class VisualizationData:
    request: PlotRequest
    fields: tuple
    bodies: tuple
    center: np.ndarray
    radius: float
    time: float
    message: str = ''


@dataclass
class ProfileData:
    request: PlotRequest
    profiles: tuple
    message: str = ''


PHASE_COLUMNS = ['Phase', 'Timestep', 'Max_2D', 'Mean_2D', 'Max_3D', 'Mean_3D']


@dataclass
class PhaseData:
    request: PlotRequest
    frame: pd.DataFrame


def validate_phase_frame(frame):
    if not set(PHASE_COLUMNS).issubset(frame.columns):
        raise ValueError('Phase CSV must contain: ' + ', '.join(PHASE_COLUMNS))
    frame = frame[PHASE_COLUMNS].apply(pd.to_numeric, errors='raise').copy()
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Phase CSV contains nonfinite values')
    return frame


class VisualizationDataProvider:
    """Bounded, process-local cache; external analyzer mutation is unsupported.

    Every caller receives copied arrays, never the cache's or analyzer's arrays.
    Display settings deliberately do not participate in scientific cache keys.
    """

    def __init__(self, analyzer, cache_size=8):
        if cache_size < 1:
            raise ValueError('cache_size must be positive')
        self.analyzer = analyzer
        self.cache_size = cache_size
        self._cache = OrderedDict()
        if not hasattr(analyzer, '_visualization_lock'):
            analyzer._visualization_lock = threading.RLock()
        self._lock = analyzer._visualization_lock
        self._configuration = None
        self._directory = os.getcwd()

    def _signature(self):
        a = self.analyzer
        state = (a.reference_system, a.cutoffs, GLOBAL_PARAMETERS.params,
                 getattr(a, 'source_parameter_sets', []))
        return hashlib.sha256(pickle.dumps(state)).digest()

    def refresh(self, reload_archive=True):
        """Explicitly reload an archive after on-disk simulation data changes."""
        with self._lock:
            self._check_directory()
            self._cache.clear()
            self._configuration = None
            a = self.analyzer
            a.cached_timestep = None
            if reload_archive:
                from src import grain_analysis
                a._load_parameters()
                a.sa = a._load_simulation_archive()
                a.metadata_version, a.grain_populations, a._grain_records = (
                    grain_analysis.read_grain_archive('simdata/particle_params.h5'))

    def _check_directory(self):
        if os.getcwd() != self._directory:
            raise ValueError('The dataset directory changed. Launch a new process for another dataset.')

    def prepare(self, request):
        with self._lock:
            self._check_directory()
            if request.view not in ('planar', 'los', '3d', 'profile', 'phase'):
                raise ValueError('Choose planar, los, 3d, profile or phase view')
            if request.kind not in ('gas', 'grain', 'mixed'):
                raise ValueError('Choose gas, grain or mixed fields')
            if request.kind != 'gas' and request.view in ('profile', 'phase'):
                raise ValueError('Grain profile cuts and phase curves are not supported')
            if request.kind != 'gas' and request.view == 'los' and request.dimension != 3:
                raise ValueError('Grain LOS requires projected 3D density, not an xy column field')
            if request.dimension not in (2, 3):
                raise ValueError('Estimator dimension must be 2 or 3')
            if request.view == '3d' and request.dimension != 3:
                raise ValueError('3D requires a volume density (dimension=3)')
            if isinstance(request.timestep, bool) or not isinstance(request.timestep, (int, np.integer)):
                raise ValueError('Timestep must be an integer archive index')
            if not 0 <= request.timestep < len(self.analyzer.sa):
                raise ValueError('Timestep is outside the simulation archive')
            signature = self._signature()
            if signature != self._configuration:
                self._cache.clear()
                self.analyzer.cached_timestep = None
                self._configuration = signature
            file_stamp = None
            if request.view == 'phase' and request.phase_file:
                stat = os.stat(request.phase_file)
                file_stamp = (stat.st_mtime_ns, stat.st_size)
            key = (id(self.analyzer.sa), signature, request, file_stamp)
            if key not in self._cache:
                self._cache[key] = self._prepare_fields(request)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return copy.deepcopy(self._cache[key])

    def _geometry(self):
        a = self.analyzer
        primary = a.get_primary(a.reference_system)
        bodies = []
        for p in a.sim.particles[:a.sim.N_active]:
            if p.r <= 0:
                continue
            orbit = None
            if p.index > 0:
                try:
                    orbit = np.asarray(p.sample_orbit(Npts=128, primary=a.get_primary(p.hash))).copy()
                except (ValueError, AttributeError):
                    pass
            bodies.append(BodyGeometry(str(p.hash.value), np.array(p.xyz), p.r,
                                       p.hash.value == primary.hash.value, p.index == 0, orbit))
        return tuple(bodies), np.array(primary.xyz), primary.r or 1.

    def _prepare_fields(self, request):
        a = self.analyzer
        if request.view == 'phase' and request.phase_file:
            return PhaseData(request, validate_phase_frame(pd.read_csv(request.phase_file)))
        a.load_timestep_data(request.timestep)
        species = GLOBAL_PARAMETERS.get('all_species', [])
        if request.species_ids is not None:
            missing = set(request.species_ids) - {s.id for s in species}
            if missing:
                raise ValueError('Selected gas species is not in this dataset')
            species = [s for s in species if s.id in request.species_ids]
        if request.view == 'phase':
            if len(species) != 1:
                raise ValueError('Select exactly one gas species for a phase curve')
            if request.source_hash is None:
                raise ValueError('Select a source for the phase curve')
            frame = a.calculate_phase_data(request.source_hash, species_name=species[0].name, orbits=request.orbits)
            return PhaseData(request, validate_phase_frame(frame))
        if request.view == 'profile':
            all_species = GLOBAL_PARAMETERS.get('all_species', [])
            profiles = tuple(a.calculate_profile(request.timestep, species_num=all_species.index(s) + 1,
                                                 max_distance_rp=request.max_distance_rp,
                                                 source_hash=request.source_hash) for s in species)
            return ProfileData(request, profiles, '' if profiles else 'No gas species selected or available.')
        dimension = 2 if request.view == 'los' else request.dimension
        fields = []
        for s in species if request.kind != 'grain' else []:
            positions = a.particle_positions[a.particle_species == s.id].copy()
            density, triangulation = a.delaunay_field_estimation(
                request.timestep, s, d=dimension, los=request.view == 'los')
            density = np.asarray(density).copy()
            if len(density) != len(positions):
                raise ValueError('Density and particle coordinates are not aligned')
            valid = np.isfinite(density) & (density > 0) & np.all(np.isfinite(positions), axis=1)
            visible = a._los_visibility_mask(positions) if request.view == 'los' else np.ones(len(density), bool)
            simplices = (np.empty((0, dimension + 1), dtype=int) if triangulation is None
                         else triangulation.simplices.copy())
            fields.append(FieldData(str(s.id), s.name, positions, density, valid, visible,
                                    simplices, 'cm^-2' if dimension == 2 else 'cm^-3',
                                    message='' if valid.any() else 'No finite positive density for this selection.'))
        if request.kind != 'gas':
            populations = request.population_ids if request.population_ids is not None else (None,)
            source_hash = request.source_hash
            if source_hash is not None:
                source_hash = int(source_hash) if str(source_hash).isdigit() else a.sim.particles[source_hash].hash.value
            for population in populations:
                result = a.grain_field(request.timestep, quantity=request.quantity, d=request.dimension,
                                       population_id=population, source_hash=source_hash,
                                       radius_range=request.radius_range)
                positions = result['positions'].copy()
                if positions.shape[1] == 2:
                    positions = np.column_stack((positions, np.zeros(len(positions))))
                density = result['density'].copy()
                valid = np.isfinite(density) & (density > 0) & np.isfinite(positions).all(axis=1)
                estimator = result['estimator']
                simplices = (np.empty((0, request.dimension + 1), int) if estimator is None
                             else estimator.delaunay.simplices.copy())
                label = f'Grain {population if population is not None else "all"} · {request.quantity}'
                if request.view == 'los':
                    label += ' · projected 3D density (not column)'
                if request.quantity == 'area':
                    label += ' · geometric area, not optical depth'
                fields.append(FieldData(f'grain:{population}', label, positions, density, valid,
                                        np.ones(len(density), bool), simplices, result['units'], request.quantity,
                                        '' if valid.any() else 'No finite positive grain density.'))
        bodies, center, radius = self._geometry()
        return VisualizationData(request, tuple(fields), bodies, center, radius, a.sim.t,
                                 '' if fields else 'No gas species selected or available.')