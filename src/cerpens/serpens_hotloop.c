/*
 * serpens_hotloop.c
 *
 * Drop-in C replacement for SerpensSimulation.advance_integrate().
 * Implements:
 *   - Magnetic dipole Lorentz force (additional_forces callback)
 *   - Threaded split-integrate-merge of test particles via pthreads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "rebound.h"
#include "reboundx.h"

/* ========================================================================
 *  Small vector helpers
 * ======================================================================== */

static inline double v3_dot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void v3_cross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline double v3_norm(const double a[3])
{
    return sqrt(v3_dot(a, a));
}

/* Rotate v by `angle` about the unit axis khat (Rodrigues' rotation formula). */
static void v3_rotate(const double khat[3], double angle, const double v[3], double out[3])
{
    const double c = cos(angle);
    const double s = sin(angle);
    const double kdotv = v3_dot(khat, v);
    double kcrossv[3];
    v3_cross(khat, v, kcrossv);

    for (int i = 0; i < 3; i++) {
        out[i] = v[i]*c + kcrossv[i]*s + khat[i]*kdotv*(1.0 - c);
    }
}

/* ========================================================================
 *  Lorentz force configuration  (set from Python before calling integrate)
 * ======================================================================== */

typedef struct {
    int    enabled;
    int    central_index;       /* index of the central body in sim->particles */
    double moment_tilted[3];    /* dipole moment with the tilt applied [A m^2] */
    double spin_hat[3];         /* unit rotation axis of the magnetosphere */
    double spin_rate;           /* |mag_rotation| [rad/s] */
    double rotation[3];         /* full rotation vector Omega [rad/s] */
    double softening;           /* ignore r < softening [m] */
} lorentz_config_t;

static lorentz_config_t g_lorentz = {0};

/* Unit vector perpendicular to n. Which perpendicular direction we pick only sets the
 * zero point of the rotation phase, so we take the one that is numerically best
 * conditioned (cross with the least-aligned coordinate axis). */
static void v3_perpendicular(const double n[3], double out[3])
{
    double axis[3] = {0.0, 0.0, 0.0};
    int least = 0;
    double smallest = fabs(n[0]);
    if (fabs(n[1]) < smallest) { least = 1; smallest = fabs(n[1]); }
    if (fabs(n[2]) < smallest) { least = 2; }
    axis[least] = 1.0;

    v3_cross(n, axis, out);
    double norm = v3_norm(out);
    if (norm == 0.0) {
        out[0] = 1.0; out[1] = 0.0; out[2] = 0.0;
        return;
    }
    out[0] /= norm; out[1] /= norm; out[2] /= norm;
}

/* Exported: Python sets these before each integrate call */
void serpens_set_lorentz_config(
    int enabled,
    int central_index,
    double mx, double my, double mz,
    double mag_tilt_rad,
    double rotx, double roty, double rotz,
    double softening)
{
    memset(&g_lorentz, 0, sizeof(g_lorentz));

    g_lorentz.enabled        = enabled;
    g_lorentz.central_index  = central_index;
    g_lorentz.softening      = softening;
    g_lorentz.rotation[0]    = rotx;
    g_lorentz.rotation[1]    = roty;
    g_lorentz.rotation[2]    = rotz;
    g_lorentz.spin_rate      = sqrt(rotx*rotx + roty*roty + rotz*rotz);

    if (g_lorentz.spin_rate > 0.0) {
        g_lorentz.spin_hat[0] = rotx / g_lorentz.spin_rate;
        g_lorentz.spin_hat[1] = roty / g_lorentz.spin_rate;
        g_lorentz.spin_hat[2] = rotz / g_lorentz.spin_rate;
    } else {
        g_lorentz.spin_hat[0] = 0.0;
        g_lorentz.spin_hat[1] = 0.0;
        g_lorentz.spin_hat[2] = 1.0;
    }

    /* The tilt is a genuine rotation of the moment vector away from the spin axis, not a
     * componentwise rescaling: a moment already aligned with the spin axis has to end up
     * at `mag_tilt_rad` from it, which requires rotating about a perpendicular axis. */
    const double moment[3] = {mx, my, mz};
    double tilt_axis[3];
    v3_perpendicular(g_lorentz.spin_hat, tilt_axis);
    v3_rotate(tilt_axis, mag_tilt_rad, moment, g_lorentz.moment_tilted);
}

/* Dipole moment at time t: the tilted moment carried around by the rotating planet. */
static inline void dipole_moment_at(double t, double m_out[3])
{
    v3_rotate(g_lorentz.spin_hat, g_lorentz.spin_rate * t, g_lorentz.moment_tilted, m_out);
}

/* ========================================================================
 *  Dipole B-field in SI
 * ======================================================================== */

static inline void dipole_B(const double m_vec[3], const double r_vec[3], double B_out[3])
{
    double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
    if (r2 == 0.0) { B_out[0] = B_out[1] = B_out[2] = 0.0; return; }

    double r     = sqrt(r2);
    double inv_r = 1.0 / r;
    double rhat[3] = { r_vec[0]*inv_r, r_vec[1]*inv_r, r_vec[2]*inv_r };

    double mu0_4pi = 1e-7;
    double factor  = mu0_4pi / (r * r2);   /* mu0/(4pi * r^3) */

    double m_dot_rhat = m_vec[0]*rhat[0] + m_vec[1]*rhat[1] + m_vec[2]*rhat[2];

    B_out[0] = factor * (3.0 * rhat[0] * m_dot_rhat - m_vec[0]);
    B_out[1] = factor * (3.0 * rhat[1] * m_dot_rhat - m_vec[1]);
    B_out[2] = factor * (3.0 * rhat[2] * m_dot_rhat - m_vec[2]);
}

/* ========================================================================
 *  REBOUND additional_forces callback — Lorentz force
 * ======================================================================== */

static void lorentz_force(struct reb_simulation* sim)
{
    if (!g_lorentz.enabled) return;

    int ci = g_lorentz.central_index;
    if (ci < 0 || (uint32_t)ci >= sim->N) return;

    struct reb_particle* central = &sim->particles[ci];

    /* Time-dependent dipole orientation */
    double m_vec[3];
    dipole_moment_at(sim->t, m_vec);

    double soft2 = g_lorentz.softening * g_lorentz.softening;

    struct rebx_extras* rebx = sim->extras;  /* rebx attached to this sim */

    for (uint32_t i = sim->N_active; i < sim->N; i++) {
        struct reb_particle* p = &sim->particles[i];

        /* Get q/m from REBOUNDx params */
        double* q_over_m_ptr = rebx_get_param(rebx, p->ap, "q_over_m");
        if (q_over_m_ptr == NULL) continue;
        double qm = *q_over_m_ptr;
        if (qm == 0.0) continue;

        double r_rel[3] = {
            p->x - central->x,
            p->y - central->y,
            p->z - central->z
        };

        if (soft2 > 0.0) {
            double rr = r_rel[0]*r_rel[0] + r_rel[1]*r_rel[1] + r_rel[2]*r_rel[2];
            if (rr < soft2) continue;
        }

        double v_rel[3] = {
            p->vx - central->vx,
            p->vy - central->vy,
            p->vz - central->vz
        };

        double B[3];
        dipole_B(m_vec, r_rel, B);


        /* v_eff = v_rel - (omega x r_rel)   (corotation correction) */
        double* rot = g_lorentz.rotation;
        double omega_cross_r[3] = {
            rot[1]*r_rel[2] - rot[2]*r_rel[1],
            rot[2]*r_rel[0] - rot[0]*r_rel[2],
            rot[0]*r_rel[1] - rot[1]*r_rel[0]
        };
        double v_eff[3] = {
            v_rel[0] - omega_cross_r[0],
            v_rel[1] - omega_cross_r[1],
            v_rel[2] - omega_cross_r[2]
        };

        /* a = (q/m) * (v_eff x B) */
        p->ax += qm * (v_eff[1]*B[2] - v_eff[2]*B[1]);
        p->ay += qm * (v_eff[2]*B[0] - v_eff[0]*B[2]);
        p->az += qm * (v_eff[0]*B[1] - v_eff[1]*B[0]);
    }
}

/* ========================================================================
 *  Thread worker for parallel integration
 * ======================================================================== */

typedef struct {
    struct reb_simulation*  sim;
    struct rebx_extras*     rebx;
    double                  target_time;
    int                     fix_circular;
} worker_arg_t;

/* Heartbeat: fix source orbit to circular */
static void heartbeat_c(struct reb_simulation* sim)
{
    /* Iterate over sources.
     * In the threaded copies, sources are still present as active particles.
     * We look for particles with source_primary set via REBOUNDx. */
    struct rebx_extras* rebx = sim->extras;
    for (int i = 0; i < sim->N_active; i++) {
        struct reb_particle* source = &sim->particles[i];
        uint32_t* sp_hash_ptr = rebx_get_param(rebx, source->ap, "source_primary");
        if (sp_hash_ptr == NULL) continue;

        /* Find primary by hash */
        struct reb_particle* primary = NULL;
        for (int j = 0; j < sim->N_active; j++) {
            if (sim->particles[j].hash == *sp_hash_ptr) {
                primary = &sim->particles[j];
                break;
            }
        }
        if (primary == NULL) continue;

        struct reb_orbit o = reb_orbit_from_particle(sim->G, *source, *primary);
        struct reb_particle newP = reb_particle_from_orbit(
            sim->G, *primary, source->m, o.a, 0.0, o.inc, o.Omega, o.omega, o.f);

        source->x  = newP.x;  source->y  = newP.y;  source->z  = newP.z;
        source->vx = newP.vx; source->vy = newP.vy; source->vz = newP.vz;
    }
}

static void* worker_thread(void* arg)
{
    worker_arg_t* w = (worker_arg_t*)arg;

    if (w->fix_circular) {
        w->sim->heartbeat = heartbeat_c;
    }

    /* Attach additional forces */
    w->sim->additional_forces          = lorentz_force;
    w->sim->force_is_velocity_dependent = 1;

    reb_simulation_integrate(w->sim, w->target_time);
    return NULL;
}


/* ========================================================================
 *  Main exported function: serpens_advance_integrate
 *
 *  Called from Python. Receives flat arrays describing the simulation state,
 *  performs the split-integrate-merge, and writes results back.
 *
 *  Arguments:
 *    n_active        – number of active (gravitating) particles
 *    n_total         – total number of particles
 *    state_in        – flat array [n_total * 7]: m, x, y, z, vx, vy, vz per particle
 *    hashes_in       – uint32 array [n_total]
 *    beta_values     – double array [n_total]  (radiation_forces beta, 0 for active)
 *    qm_values       – double array [n_total]  (q/m values, 0 for active)
 *    rad_source_flags– int array [n_total]     (1 if radiation source, 0 otherwise)
 *    target_time     – integration target time (absolute)
 *    G_value         – gravitational constant
 *    min_dt          – minimum timestep for IAS15
 *    n_threads       – number of threads to use
 *    fix_circular    – 1 if heartbeat should fix circular orbits
 *    state_out       – output array, same layout as state_in (preallocated)
 *    n_out           – output: actual number of particles after merges/collisions
 *    sim_time_out    – output: final simulation time
 * ======================================================================== */

void serpens_advance_integrate(
    int n_active,
    int n_total,
    const double* state_in,
    const uint32_t* hashes_in,
    const double* beta_values,
    const double* qm_values,
    const int* rad_source_flags,
    const uint32_t* source_primary_hashes,
    double target_time,
    double G_value,
    double min_dt,
    double sim_t0,
    int n_threads,
    int fix_circular,
    /* outputs */
    double* state_out,
    uint32_t* hashes_out,
    int* n_out,
    double* sim_time_out)
{
    if (n_threads < 1) n_threads = 1;
    int n_test = n_total - n_active;
    if (n_threads > n_test && n_test > 0) n_threads = n_test;
    if (n_test == 0) n_threads = 1;

    /* --- Allocate worker data --- */
    worker_arg_t* workers = (worker_arg_t*)calloc(n_threads, sizeof(worker_arg_t));
    pthread_t*    threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));

    /* Compute how many test particles per thread */
    int base_count = n_test / n_threads;
    int remainder  = n_test % n_threads;

    int test_offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int my_test_count = base_count + (t < remainder ? 1 : 0);

        /* Create simulation */
        struct reb_simulation* sim = reb_simulation_create();
        sim->G = G_value;
        sim->integrator = REB_INTEGRATOR_IAS15;
        sim->ri_ias15.min_dt = min_dt;
        sim->collision = REB_COLLISION_DIRECT;
        sim->collision_resolve = reb_collision_resolve_merge;
        sim->t = sim_t0;

        /* Add active particles */
        for (int i = 0; i < n_active; i++) {
            struct reb_particle p = {0};
            p.m  = state_in[i*7 + 0];
            p.x  = state_in[i*7 + 1];
            p.y  = state_in[i*7 + 2];
            p.z  = state_in[i*7 + 3];
            p.vx = state_in[i*7 + 4];
            p.vy = state_in[i*7 + 5];
            p.vz = state_in[i*7 + 6];
            p.hash = hashes_in[i];
            reb_simulation_add(sim, p);
        }

        /* Add this thread's slice of test particles */
        for (int j = 0; j < my_test_count; j++) {
            int gi = n_active + test_offset + j;  /* global index */
            struct reb_particle p = {0};
            p.m  = state_in[gi*7 + 0];
            p.x  = state_in[gi*7 + 1];
            p.y  = state_in[gi*7 + 2];
            p.z  = state_in[gi*7 + 3];
            p.vx = state_in[gi*7 + 4];
            p.vy = state_in[gi*7 + 5];
            p.vz = state_in[gi*7 + 6];
            p.hash = hashes_in[gi];
            reb_simulation_add(sim, p);
        }

        sim->N_active = n_active;

        /* Attach REBOUNDx and set per-particle parameters */
        struct rebx_extras* rebx = rebx_attach(sim);

        /* Register custom parameters */
        rebx_register_param(rebx, "q_over_m", REBX_TYPE_DOUBLE);
        rebx_register_param(rebx, "source_primary", REBX_TYPE_UINT32);

        /* Radiation forces */
        struct rebx_force* rf = rebx_load_force(rebx, "radiation_forces");
        rebx_add_force(rebx, rf);
        rebx_set_param_double(rebx, &rf->ap, "c", 3.0e8);

        /* Active particles: set radiation_source flag and source_primary */
        for (int i = 0; i < n_active; i++) {
            if (rad_source_flags[i]) {
                rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[i].ap, "radiation_source", 1);
            }
            /* beta and q/m for active particles (usually 0) */
            if (beta_values[i] != 0.0) {
                rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[i].ap, "beta", beta_values[i]);
            }
            if (qm_values[i] != 0.0) {
                rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[i].ap, "q_over_m", qm_values[i]);
            }
            /* source_primary hash for heartbeat circular orbit fixing */
            if (source_primary_hashes[i] != 0) {
                rebx_set_param_uint32(rebx, (struct rebx_node**)&sim->particles[i].ap, "source_primary", source_primary_hashes[i]);
            }
        }

        /* Test particles: set beta and q_over_m */
        for (int j = 0; j < my_test_count; j++) {
            int gi = n_active + test_offset + j;
            int li = n_active + j;  /* local index in this sim */
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "beta", beta_values[gi]);
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "q_over_m", qm_values[gi]);
        }

        workers[t].sim          = sim;
        workers[t].rebx         = rebx;
        workers[t].target_time  = target_time;
        workers[t].fix_circular = fix_circular;

        test_offset += my_test_count;
    }

    /* --- Launch threads --- */
    for (int t = 0; t < n_threads; t++) {
        pthread_create(&threads[t], NULL, worker_thread, &workers[t]);
    }

    /* --- Join threads --- */
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    /* --- Merge results back --- */
    /* Active particles come from thread 0 (they should all agree) */
    int out_idx = 0;

    for (int i = 0; i < workers[0].sim->N_active; i++) {
        struct reb_particle* p = &workers[0].sim->particles[i];
        state_out[out_idx*7 + 0] = p->m;
        state_out[out_idx*7 + 1] = p->x;
        state_out[out_idx*7 + 2] = p->y;
        state_out[out_idx*7 + 3] = p->z;
        state_out[out_idx*7 + 4] = p->vx;
        state_out[out_idx*7 + 5] = p->vy;
        state_out[out_idx*7 + 6] = p->vz;
        hashes_out[out_idx] = p->hash;
        out_idx++;
    }

    /* Test particles from each thread */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        for (uint32_t i = sim->N_active; i < sim->N; i++) {
            struct reb_particle* p = &sim->particles[i];
            state_out[out_idx*7 + 0] = p->m;
            state_out[out_idx*7 + 1] = p->x;
            state_out[out_idx*7 + 2] = p->y;
            state_out[out_idx*7 + 3] = p->z;
            state_out[out_idx*7 + 4] = p->vx;
            state_out[out_idx*7 + 5] = p->vy;
            state_out[out_idx*7 + 6] = p->vz;
            hashes_out[out_idx] = p->hash;
            out_idx++;
        }
    }

    *n_out = out_idx;
    *sim_time_out = workers[0].sim->t;

    /* --- Cleanup --- */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        struct rebx_extras* rebx = workers[t].rebx;

        if (rebx && sim) {
            rebx_detach(sim, rebx);
            workers[t].rebx = NULL;
        }

        if (sim) {
            reb_simulation_free(sim);
            workers[t].sim = NULL;
        }
    }
    free(workers);
    free(threads);
}

