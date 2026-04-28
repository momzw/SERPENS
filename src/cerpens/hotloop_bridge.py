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

# void serpens_advance_integrate(int, int, double*, uint32*, double*, double*,
#                                int*, uint32*, double, double, double, double,
#                                int, int, double*, uint32*, int*, double*)
_lib.serpens_advance_integrate.restype = None
_lib.serpens_advance_integrate.argtypes = [
    ctypes.c_int,                                   # n_active
    ctypes.c_int,                                   # n_total
    ctypes.POINTER(ctypes.c_double),                # state_in
    ctypes.POINTER(ctypes.c_uint32),                # hashes_in
    ctypes.POINTER(ctypes.c_double),                # beta_values
    ctypes.POINTER(ctypes.c_double),                # qm_values
    ctypes.POINTER(ctypes.c_int),                   # rad_source_flags
    ctypes.POINTER(ctypes.c_uint32),                # source_primary_hashes
    ctypes.c_double,                                # target_time
    ctypes.c_double,                                # G_value
    ctypes.c_double,                                # min_dt
    ctypes.c_double,                                # sim_t0
    ctypes.c_int,                                   # n_threads
    ctypes.c_int,                                   # fix_circular
    ctypes.POINTER(ctypes.c_double),                # state_out
    ctypes.POINTER(ctypes.c_uint32),                # hashes_out
    ctypes.POINTER(ctypes.c_int),                   # n_out
    ctypes.POINTER(ctypes.c_double),                # sim_time_out
]

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
        Whether to fix source orbits to circular.
    params : Parameters
        GLOBAL_PARAMETERS instance for Lorentz config.
    """

    n_active = sim.N_active if sim.N_active >= 0 else sim.N
    n_total = sim.N

    # Snapshot radii for active particles — not part of the 7-element state vector
    active_radii = [sim.particles[i].r for i in range(n_active)]

    # --- Pack simulation state into flat arrays ---
    state_in = np.zeros(n_total * 7, dtype=np.float64)
    hashes_in = np.zeros(n_total, dtype=np.uint32)
    beta_values = np.zeros(n_total, dtype=np.float64)
    qm_values = np.zeros(n_total, dtype=np.float64)
    rad_source_flags = np.zeros(n_total, dtype=np.int32)
    source_primary_hashes = np.zeros(n_total, dtype=np.uint32)

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

        # REBOUNDx parameters
        try:
            beta_val = p.params.get("beta")
            if beta_val is not None:
                beta_values[i] = float(beta_val)
        except AttributeError:
            pass

        try:
            qm_val = p.params.get("q_over_m")
            if qm_val is not None:
                qm_values[i] = float(qm_val)
        except AttributeError:
            pass

        try:
            rs_val = p.params.get("radiation_source")
            if rs_val is not None:
                rad_source_flags[i] = int(rs_val)
        except AttributeError:
            pass

        # source_primary from h5 (for active particles used by heartbeat_c)
        if i < n_active:
            sp = sim.get_particle_param(p.hash.value, "source_primary")
            if sp is not None:
                source_primary_hashes[i] = int(sp)

    # --- Configure Lorentz force in C ---
    configure_lorentz(params)

    # --- Prepare output buffers ---
    state_out = np.zeros(n_total * 7, dtype=np.float64)
    hashes_out = np.zeros(n_total, dtype=np.uint32)
    n_out = ctypes.c_int(0)
    sim_time_out = ctypes.c_double(0.0)

    # --- Call C function ---
    _lib.serpens_advance_integrate(
        ctypes.c_int(n_active),
        ctypes.c_int(n_total),
        state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hashes_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        beta_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        qm_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rad_source_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        source_primary_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_double(target_time),
        ctypes.c_double(sim.G),
        ctypes.c_double(1e-3),
        ctypes.c_double(sim.t),
        ctypes.c_int(n_threads),
        ctypes.c_int(int(fix_circular)),
        state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hashes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(n_out),
        ctypes.byref(sim_time_out),
    )

    # --- Unpack results back into the simulation ---
    actual_n = n_out.value
    actual_n_active = min(n_active, actual_n)

    # Clear existing particles
    while sim.N > 0:
        sim.remove(index=sim.N - 1)

    # Re-add active particles
    import rebound
    for i in range(actual_n_active):
        p = rebound.Particle()
        p.m  = state_out[i*7 + 0]
        p.x  = state_out[i*7 + 1]
        p.y  = state_out[i*7 + 2]
        p.z  = state_out[i*7 + 3]
        p.vx = state_out[i*7 + 4]
        p.vy = state_out[i*7 + 5]
        p.vz = state_out[i*7 + 6]
        p.hash = rebound.hash(int(hashes_out[i]))
        rebound.Simulation.add(sim, p)

    # Re-add test particles
    for i in range(actual_n_active, actual_n):
        p = rebound.Particle()
        p.m  = state_out[i*7 + 0]
        p.x  = state_out[i*7 + 1]
        p.y  = state_out[i*7 + 2]
        p.z  = state_out[i*7 + 3]
        p.vx = state_out[i*7 + 4]
        p.vy = state_out[i*7 + 5]
        p.vz = state_out[i*7 + 6]
        p.hash = rebound.hash(int(hashes_out[i]))
        rebound.Simulation.add(sim, p)

    sim.N_active = actual_n_active
    sim.t = sim_time_out.value

    # Restore particle radii for active particles (not in state vector)
    for i in range(actual_n_active):
        if i < len(active_radii):
            sim.particles[i].r = active_radii[i]

    # Restore per-particle params
    for i in range(actual_n_active, actual_n):
        # Restore REBOUNDx params
        #sim.particles[-1].params["beta"] = beta_values[i] if i < len(beta_values) else 0.0
        sim.particles[i].params["q_over_m"] = float(qm_values[i]) if i < len(qm_values) else 0.0

