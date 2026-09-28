"""
Bridge between the SERPENS Python simulation and the C hot loop.
Loads serpens_hotloop.so/.dll and provides a drop-in replacement
for SerpensSimulation.advance_integrate().
"""

import ctypes
import os
import platform

import numpy as np
import rebound
import reboundx


# --- Load dependencies into the GLOBAL namespace first ---
def load_dependency_globally(package, prefix):
    """
    Search for a shared library within the package OR its parent directory
    (site-packages) and load it globally.
    """
    if not hasattr(package, "__path__") or not package.__path__:
        raise ImportError(f"Module '{package.__name__}' has no path. Is it installed correctly?")

    # 1. The package directory (e.g., .../site-packages/rebound)
    pkg_path = package.__path__[0]
    # 2. The parent directory (e.g., .../site-packages/)
    parent_path = os.path.dirname(pkg_path)

    # We will search both locations
    search_dirs = [pkg_path, parent_path]

    for search_dir in search_dirs:
        # We don't necessarily need os.walk here if it's just in site-packages,
        # but it doesn't hurt to keep it for nested builds.
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.startswith(prefix) and (f.endswith(".so") or f.endswith(".dylib")):
                    lib_full_path = os.path.join(root, f)
                    try:
                        #print(f"DEBUG: Found library at {lib_full_path}")
                        return ctypes.CDLL(lib_full_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError as e:
                        raise ImportError(f"Failed to load {lib_full_path}: {e}")

    raise ImportError(
        f"Could not find '{prefix}' in {pkg_path} or its parent {parent_path}. "
        "Check your site-packages for the .so file."
    )


# Load REBOUND and REBOUNDx libraries globally
try:
    _lib_rebound = load_dependency_globally(rebound, "librebound")
    _lib_reboundx = load_dependency_globally(reboundx, "libreboundx")
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    # Exit or handle gracefully
    raise e

# --- Load shared library ---
_dir = os.path.dirname(os.path.abspath(__file__))
_ext = ".dll" if platform.system() == "Windows" else ".so"
_lib_path = os.path.join(_dir, f"serpens_hotloop{_ext}")
_lib = ctypes.CDLL(_lib_path)

# --- Check the shared library matches this module ---
# serpens_hotloop.so is not checked in, so a stale build left over from before an argument
# list changed would be handed the wrong arguments and silently read garbage.
_REQUIRED_ABI = 2
try:
    _lib.serpens_hotloop_abi_version.restype = ctypes.c_int
    _found_abi = _lib.serpens_hotloop_abi_version()
except AttributeError:
    _found_abi = 0
if _found_abi != _REQUIRED_ABI:
    raise ImportError(
        f"{_lib_path} is out of date (ABI {_found_abi}, expected {_REQUIRED_ABI}). "
        "Rebuild it with `make` in src/cerpens/."
    )

# --- Declare function signatures ---

# void serpens_set_lorentz_config(int, int, double, ..., double)
_lib.serpens_set_lorentz_config.restype = None
_lib.serpens_set_lorentz_config.argtypes = [
    ctypes.c_int,       # enabled
    ctypes.c_int,       # central_index
    ctypes.c_double,    # mx
    ctypes.c_double,    # my
    ctypes.c_double,    # mz
    ctypes.c_double,    # mag_tilt_rad
    ctypes.c_double,    # rotx
    ctypes.c_double,    # roty
    ctypes.c_double,    # rotz
    ctypes.c_double,    # softening
]

_lib.serpens_advance_integrate.restype = ctypes.c_int
_lib.serpens_advance_integrate.argtypes = [
    ctypes.c_int,                                   # n_active
    ctypes.c_int,                                   # n_total
    ctypes.POINTER(ctypes.c_double),                # state_in
    ctypes.POINTER(ctypes.c_uint32),                # hashes_in
    ctypes.POINTER(ctypes.c_double),                # beta_values
    ctypes.POINTER(ctypes.c_double),                # qm_values
    ctypes.POINTER(ctypes.c_double),                # mu_values
    ctypes.POINTER(ctypes.c_double),                # radii
    ctypes.POINTER(ctypes.c_int),                   # rad_source_flags
    ctypes.POINTER(ctypes.c_uint32),                # source_primary_hashes
    ctypes.POINTER(ctypes.c_double),                # radii_in
    ctypes.POINTER(ctypes.c_int),                   # particle_kinds
    ctypes.c_double,                                # target_time
    ctypes.c_double,                                # G_value
    ctypes.c_double,                                # min_dt
    ctypes.c_double,                                # gc_rtol
    ctypes.c_double,                                # gc_eps_max
    ctypes.c_double,                                # epsilon
    ctypes.c_double,                                # sim_t0
    ctypes.c_double,                                # initial_dt
    ctypes.c_double,                                # max_dt (0 disables)
    ctypes.c_double,                                # radiation_c
    ctypes.c_int,                                   # n_threads
    ctypes.c_int,                                   # force_is_velocity_dependent
    ctypes.POINTER(ctypes.c_double),                # state_out
    ctypes.POINTER(ctypes.c_uint32),                # hashes_out
    ctypes.POINTER(ctypes.c_double),                # qm_out
    ctypes.POINTER(ctypes.c_double),                # mu_out
    ctypes.POINTER(ctypes.c_double),                # radii_out
    ctypes.POINTER(ctypes.c_int),                   # n_out
    ctypes.POINTER(ctypes.c_int),                   # n_active_out
    ctypes.POINTER(ctypes.c_double),                # sim_time_out
]

# Uninitialised magnetic moment: the C side derives mu/m from the particle's velocity the
# first time it sees a charged particle, then carries it as an adiabatic invariant.
MU_UNSET = -1.0

# Defaults for the integration controls that used to be hardcoded here.
DEFAULT_IAS15_MIN_DT = 1e-4
DEFAULT_GC_RTOL = 1e-6

# Validity threshold of the guiding-centre expansion, eps = r_gyro / (B/|grad B|). Above
# it a particle is no longer well magnetised and is integrated ballistically instead. This
# is a conventional cutoff on the expansion parameter, not a measured quantity.
DEFAULT_GC_EPS_MAX = 0.1

def configure_lorentz(params):
    """Push Lorentz config from GLOBAL_PARAMETERS into the C library."""
    enabled = int(params.get("lorentz_enabled", False))
    if not enabled:
        _lib.serpens_set_lorentz_config(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return

    # Central body index — caller must resolve name -> index before calling
    central_idx = int(params.get("lorentz_central_index", 1))
    m = params.get("lorentz_moment_Am2", [0.0, 0.0, 0.0])
    tilt = float(np.deg2rad(params.get("magnetic_tilt_degrees", 0.0)))
    rot = params.get("magnetic_rotation", [0.0, 0.0, 0.0])
    soft = float(params.get("lorentz_softening_m", 0.0))

    _lib.serpens_set_lorentz_config(
        1, central_idx,
        float(m[0]), float(m[1]), float(m[2]),
        tilt,
        float(rot[0]), float(rot[1]), float(rot[2]),
        soft,
    )


def advance_integrate_c(sim, target_time, n_threads, fix_circular, params):
    """
    Drop-in replacement for SerpensSimulation.advance_integrate().

    Parameters
    ----------
    sim : SerpensSimulation
        The REBOUND simulation instance.
    target_time : float
        Absolute target integration time.
    n_threads : int
        Number of threads for parallel integration.
    fix_circular : bool
        Legacy argument, ignored. The parent applies orbit fixing once per step.
    params : Parameters
        GLOBAL_PARAMETERS instance for Lorentz config and optional positive
        integration_max_dt (seconds), limiting collision-search intervals.
    """

    n_active = sim.N_active if sim.N_active >= 0 else sim.N
    n_total = sim.N

    # Snapshot radii for active particles — not part of the 7-element state vector. They
    # are also passed into C, where they act as the absorbing surface for guiding centres.
    active_radii = [sim.particles[i].r for i in range(n_active)]
    radii = np.zeros(n_total, dtype=np.float64)
    radii[:n_active] = active_radii

    max_dt = params.get('integration_max_dt')
    if max_dt is not None and (not np.isfinite(max_dt) or max_dt <= 0):
        raise ValueError('integration_max_dt must be finite and positive')
    if not np.isfinite(target_time) or target_time < sim.t:
        raise ValueError('Integration target must be finite and not earlier than sim.t')
    if target_time == sim.t:
        return

    # --- Pack simulation state into flat arrays ---
    state_in = np.zeros(n_total * 7, dtype=np.float64)
    hashes_in = np.zeros(n_total, dtype=np.uint32)
    beta_values = np.zeros(n_total, dtype=np.float64)
    qm_values = np.zeros(n_total, dtype=np.float64)
    mu_values = np.full(n_total, MU_UNSET, dtype=np.float64)
    rad_source_flags = np.zeros(n_total, dtype=np.int32)
    source_primary_hashes = np.zeros(n_total, dtype=np.uint32)
    radii_in = np.zeros(n_total, dtype=np.float64)
    particle_kinds = np.zeros(n_total, dtype=np.int32)
    metadata_by_hash = {}

    for i in range(n_total):
        p = sim.particles[i]
        state_in[i*7 + 0] = p.m
        state_in[i*7 + 1] = p.x
        state_in[i*7 + 2] = p.y
        state_in[i*7 + 3] = p.z
        state_in[i*7 + 4] = p.vx
        state_in[i*7 + 5] = p.vy
        state_in[i*7 + 6] = p.vz
        hashes_in[i] = p.hash.value
        radii_in[i] = p.r
        if p.hash.value in metadata_by_hash:
            raise ValueError('C backend requires unique particle hashes')
        metadata = {}
        for field in ('beta', 'q_over_m', 'mu_over_m', 'radiation_source', 'source_primary', 'source_hash', 'particle_kind'):
            try:
                metadata[field] = p.params[field]
            except AttributeError:
                pass
        if not metadata.get('radiation_source'):
            metadata.pop('radiation_source', None)
        metadata_by_hash[p.hash.value] = metadata
        beta_values[i] = metadata.get('beta', 0.)
        qm_values[i] = metadata.get('q_over_m', 0.)
        mu_values[i] = metadata.get('mu_over_m', 0.)
        rad_source_flags[i] = metadata.get('radiation_source', 0)
        source_primary_hashes[i] = sim.get_particle_param(p.hash.value, 'source_primary') or 0
        particle_kinds[i] = sim.get_particle_param(p.hash.value, 'particle_kind') or metadata.get('particle_kind', 0)

    # --- Configure Lorentz force in C ---
    configure_lorentz(params)

    # --- Prepare output buffers ---
    state_out = np.zeros(n_total * 7, dtype=np.float64)
    hashes_out = np.zeros(n_total, dtype=np.uint32)
    qm_out = np.zeros(n_total, dtype=np.float64)
    mu_out = np.full(n_total, MU_UNSET, dtype=np.float64)
    radii_out = np.zeros(n_total, dtype=np.float64)
    n_out = ctypes.c_int(0)
    n_active_out = ctypes.c_int(0)
    sim_time_out = ctypes.c_double(0.0)

    min_dt = float(params.get("ias15_min_dt", DEFAULT_IAS15_MIN_DT))
    gc_rtol = float(params.get("guiding_centre_rtol", DEFAULT_GC_RTOL))
    gc_eps_max = float(params.get("guiding_centre_eps_max", DEFAULT_GC_EPS_MAX))

    # --- Call C function ---
    status = _lib.serpens_advance_integrate(
        ctypes.c_int(n_active),
        ctypes.c_int(n_total),
        state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hashes_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        beta_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qm_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mu_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rad_source_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        source_primary_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        radii_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        particle_kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_double(target_time),
        ctypes.c_double(sim.G),
        ctypes.c_double(min_dt),
        ctypes.c_double(gc_rtol),
        ctypes.c_double(gc_eps_max),
        ctypes.c_double(sim.ri_ias15.epsilon),
        ctypes.c_double(sim.t),
        ctypes.c_double(sim.dt),
        ctypes.c_double(max_dt or 0.),
        ctypes.c_double(sim.rebx.get_force('radiation_forces').params['c']),
        ctypes.c_int(n_threads),
        ctypes.c_int(sim.force_is_velocity_dependent),
        state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hashes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        qm_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mu_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        radii_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(n_out),
        ctypes.byref(n_active_out),
        ctypes.byref(sim_time_out),
    )
    if status:
        raise RuntimeError(f'C backend integration failed with REBOUND status {status}')

    # --- Unpack results back into the simulation ---
    actual_n = n_out.value
    actual_n_active = n_active_out.value

    # Clear existing particles
    del sim.particles

    # Re-add particles and restore live parameters by identity, never index.
    for i in range(actual_n):
        p = rebound.Particle()
        p.m  = state_out[i*7 + 0]
        p.x  = state_out[i*7 + 1]
        p.y  = state_out[i*7 + 2]
        p.z  = state_out[i*7 + 3]
        p.vx = state_out[i*7 + 4]
        p.vy = state_out[i*7 + 5]
        p.vz = state_out[i*7 + 6]
        p.r = radii_out[i]
        p.hash = rebound.hash(int(hashes_out[i]))
        rebound.Simulation.add(sim, p)
        for field, value in metadata_by_hash[int(hashes_out[i])].items():
            sim.particles[-1].params[field] = value

    sim.N_active = actual_n_active
    sim.t = sim_time_out.value

