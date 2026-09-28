/*
 * serpens_hotloop.c
 *
 * Drop-in C replacement for SerpensSimulation.advance_integrate().
 * Implements:
 *   - Guiding-centre transport of charged test particles in a tilted, rotating
 *     magnetic dipole
 *   - Threaded split-integrate-merge of test particles via pthreads
 *
 * Why guiding centre and not a Lorentz force term?
 * ------------------------------------------------
 * For Na+ at ~1.2 R_p in a ~10 G field the gyroperiod is a few milliseconds while a
 * single sim_advance = 0.01 hr call spans ~1e4 gyroperiods. Resolving the spiral is
 * hopeless, and feeding an unresolved v x B term to IAS15 makes |v| grow without bound.
 * The gyroradius, however, is metres against R_p ~ 1e8 m, so the drift (guiding-centre)
 * approximation is excellent: we drop the gyration and evolve the centre of the orbit
 * with the magnetic moment mu = m v_perp^2 / (2B) carried as an adiabatic invariant.
 *
 * Guiding-centre motion is first order in position (dR/dt is an algebraic function of
 * the local field) and therefore cannot be written as an acceleration. REBOUND cannot
 * integrate it, so charged test particles are held outside the reb_simulation and
 * advanced by the embedded Runge-Kutta pusher below. Neutrals and the gravitating
 * bodies are integrated by REBOUND exactly as before.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "rebound.h"
#include "reboundx.h"

/* Hard cap on substeps taken by the guiding-centre pusher for a single particle over a
 * single advance call. Only reached if the adaptive controller is stuck. */
#define GC_MAX_STEPS 100000

/* Bump whenever the signature of serpens_advance_integrate changes. The .so is not
 * checked in, so a stale build would otherwise be handed the wrong arguments and read
 * garbage rather than fail. */
#define SERPENS_HOTLOOP_ABI 2

int serpens_hotloop_abi_version(void) { return SERPENS_HOTLOOP_ABI; }

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
 *  Lorentz / guiding-centre configuration  (set from Python before integrating)
 * ======================================================================== */

typedef struct {
    int    enabled;
    int    central_index;       /* index of the central body in sim->particles */
    double moment_tilted[3];    /* dipole moment with the tilt applied [A m^2] */
    double spin_hat[3];         /* unit rotation axis of the magnetosphere */
    double spin_rate;           /* |mag_rotation| [rad/s] */
    double rotation[3];         /* full rotation vector Omega [rad/s] */
    double softening;           /* Plummer softening length [m] */
} lorentz_config_t;

static lorentz_config_t g_lorentz = {0};

/* Validity threshold of the guiding-centre expansion, epsilon = r_gyro / (B/|grad B|).
 * The drift approximation is an expansion in this ratio, so it only holds where the
 * particle is well magnetised. Set per call from the parameters; the value here is the
 * fallback, not a measurement. */
static double g_gc_eps_max = 0.1;

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

/* Softened dipole. Substituting r -> s = (r^8 + a^8)^(1/8) in both the radial falloff and
 * the radial unit vector keeps B finite and smooth through r = 0, instead of cutting the
 * force off at a hard radius (which put a discontinuity in the equations of motion right
 * where particles are most strongly magnetised).
 *
 * The high exponent matters: a Plummer-style sqrt(r^2 + a^2) would suppress |B| by a
 * factor ~2 out at 1.2a, which is inside the region a torus actually occupies and would
 * artificially widen the loss cone. This form is within 1.5% of the exact dipole by 1.5a
 * and 0.2% by 2a, while still being C-infinity at the origin (r^8 is a polynomial in the
 * components). */
static void dipole_B(const double m_vec[3], const double r_vec[3], double B_out[3])
{
    const double a = g_lorentz.softening;
    const double r2 = v3_dot(r_vec, r_vec);
    const double r8 = r2*r2*r2*r2;
    const double a2 = a*a;
    const double a8 = a2*a2*a2*a2;
    const double s8 = r8 + a8;
    if (s8 <= 0.0) { B_out[0] = B_out[1] = B_out[2] = 0.0; return; }

    const double s     = pow(s8, 0.125);
    const double s2    = s * s;
    const double inv_s = 1.0 / s;
    const double n[3]  = { r_vec[0]*inv_s, r_vec[1]*inv_s, r_vec[2]*inv_s };

    const double mu0_4pi = 1e-7;
    const double factor  = mu0_4pi / (s * s2);   /* mu0/(4pi * s^3) */

    const double m_dot_n = v3_dot(m_vec, n);

    B_out[0] = factor * (3.0 * n[0] * m_dot_n - m_vec[0]);
    B_out[1] = factor * (3.0 * n[1] * m_dot_n - m_vec[1]);
    B_out[2] = factor * (3.0 * n[2] * m_dot_n - m_vec[2]);
}

/* B, |B| and grad|B| at r_vec. The softened dipole has no compact analytic gradient, so
 * grad|B| is taken by central differences on a step scaled to the local field scale
 * length; at h/s ~ 1e-4 both truncation and round-off sit near 1e-8 relative. */
static void dipole_field(const double m_vec[3], const double r_vec[3],
                         double B_out[3], double* Bmag_out, double gradB_out[3])
{
    dipole_B(m_vec, r_vec, B_out);
    *Bmag_out = v3_norm(B_out);

    const double a = g_lorentz.softening;
    const double r2 = v3_dot(r_vec, r_vec);
    const double a2 = a*a;
    const double s = pow(r2*r2*r2*r2 + a2*a2*a2*a2, 0.125);
    const double h = 1e-4 * s;
    if (h <= 0.0) { gradB_out[0] = gradB_out[1] = gradB_out[2] = 0.0; return; }

    for (int k = 0; k < 3; k++) {
        double rp[3] = { r_vec[0], r_vec[1], r_vec[2] };
        double rm[3] = { r_vec[0], r_vec[1], r_vec[2] };
        rp[k] += h;
        rm[k] -= h;

        double Bp[3], Bm[3];
        dipole_B(m_vec, rp, Bp);
        dipole_B(m_vec, rm, Bm);
        gradB_out[k] = (v3_norm(Bp) - v3_norm(Bm)) / (2.0 * h);
    }
}

/* ========================================================================
 *  Gravitating bodies along the step
 *
 *  The guiding-centre pusher takes its own substeps, so it needs the active particles
 *  at intermediate times. We snapshot them either side of the REBOUND step and use
 *  cubic Hermite interpolation, which is exact to O(h^4) because we have both position
 *  and velocity at each end.
 * ======================================================================== */

typedef struct {
    int          n_bodies;
    int          central;       /* index of the central (magnetised) body */
    double       G;
    double       t0, t1;
    const double* mass;         /* [n_bodies] */
    const double* radius;       /* [n_bodies], absorbing radius [m] */
    const int*   rad_source;    /* [n_bodies], 1 if it drives radiation pressure */
    const double* x0;           /* [n_bodies*3] */
    const double* v0;
    const double* x1;
    const double* v1;
} body_track_t;

static void bodies_at(const body_track_t* bt, double t, double* x, double* v)
{
    const double h = bt->t1 - bt->t0;

    if (h == 0.0) {
        memcpy(x, bt->x0, (size_t)bt->n_bodies * 3 * sizeof(double));
        memcpy(v, bt->v0, (size_t)bt->n_bodies * 3 * sizeof(double));
        return;
    }

    const double u  = (t - bt->t0) / h;
    const double u2 = u * u;
    const double u3 = u2 * u;

    const double h00 =  2.0*u3 - 3.0*u2 + 1.0;
    const double h10 =        u3 - 2.0*u2 + u;
    const double h01 = -2.0*u3 + 3.0*u2;
    const double h11 =        u3 -     u2;

    const double d00 =  6.0*u2 - 6.0*u;
    const double d10 =  3.0*u2 - 4.0*u + 1.0;
    const double d01 = -6.0*u2 + 6.0*u;
    const double d11 =  3.0*u2 - 2.0*u;

    for (int i = 0; i < bt->n_bodies * 3; i++) {
        x[i] = h00*bt->x0[i] + h10*h*bt->v0[i] + h01*bt->x1[i] + h11*h*bt->v1[i];
        v[i] = (d00*bt->x0[i] + d01*bt->x1[i]) / h + d10*bt->v0[i] + d11*bt->v1[i];
    }
}

/* ========================================================================
 *  Guiding-centre equations of motion
 * ======================================================================== */

typedef struct {
    double   y[4];        /* guiding centre x, y, z [m] and parallel velocity [m/s] */
    double   mu_bar;      /* mu/m = v_perp^2 / (2B) [m^2 s^-2 T^-1]; < 0 = uninitialised */
    double   q_over_m;    /* [C/kg] */
    double   beta;        /* radiation pressure / gravity ratio */
    double   mass;        /* [kg], zero for test particles */
    double   v_full[3];   /* full velocity in, guiding-centre velocity out [m/s] */
    uint32_t hash;
    int      absorbed;    /* guiding centre ended up inside a gravitating body */
} gc_ion_t;

/* Acceleration felt by the guiding centre *relative to the central body*, i.e. the
 * combination that actually pushes the particle around in the planet's frame:
 *   - gravity of every active body on the particle, minus the same sum on the planet
 *   - radiation pressure (beta times the radiation source's gravity), on the particle only
 *   - the centrifugal term of the corotating field frame, -Omega x (Omega x r)
 * The last one is the inertial term -b.(dv_E/dt) of Northrop's parallel equation for
 * rigid corotation; it is what centrifugally confines a torus to the magnetic equator.
 */
static void gc_effective_acceleration(const body_track_t* bt, const double* bx,
                                      const double pos[3], const double r_rel[3],
                                      double beta, double a_out[3])
{
    const double* cx = &bx[bt->central * 3];

    a_out[0] = a_out[1] = a_out[2] = 0.0;

    for (int j = 0; j < bt->n_bodies; j++) {
        const double mj = bt->mass[j];
        if (mj == 0.0) continue;
        const double* xj = &bx[j*3];

        double d[3] = { pos[0]-xj[0], pos[1]-xj[1], pos[2]-xj[2] };
        double d2 = v3_dot(d, d);
        if (d2 > 0.0) {
            /* Radiation pressure acts on the particle only, and scales the source's
             * gravity by beta directed outward. */
            double pull = bt->G * mj / (d2 * sqrt(d2));
            double scale = (bt->rad_source[j] && beta != 0.0) ? (beta - 1.0) : -1.0;
            a_out[0] += scale * pull * d[0];
            a_out[1] += scale * pull * d[1];
            a_out[2] += scale * pull * d[2];
        }

        if (j == bt->central) continue;

        double dc[3] = { cx[0]-xj[0], cx[1]-xj[1], cx[2]-xj[2] };
        double dc2 = v3_dot(dc, dc);
        if (dc2 > 0.0) {
            double pull = bt->G * mj / (dc2 * sqrt(dc2));
            a_out[0] += pull * dc[0];
            a_out[1] += pull * dc[1];
            a_out[2] += pull * dc[2];
        }
    }

    /* Centrifugal acceleration of the corotating frame: -Omega x (Omega x r_rel) */
    double omega_cross_r[3], omega_cross_omega_cross_r[3];
    v3_cross(g_lorentz.rotation, r_rel, omega_cross_r);
    v3_cross(g_lorentz.rotation, omega_cross_r, omega_cross_omega_cross_r);
    a_out[0] -= omega_cross_omega_cross_r[0];
    a_out[1] -= omega_cross_omega_cross_r[1];
    a_out[2] -= omega_cross_omega_cross_r[2];
}

/* Local magnetic geometry plus the corotation E x B drift at r_rel. */
static void gc_local_field(double t, const double r_rel[3],
                           double bhat[3], double* Bmag, double gradB[3], double u_E[3])
{
    double m_vec[3], B[3];
    dipole_moment_at(t, m_vec);
    dipole_field(m_vec, r_rel, B, Bmag, gradB);

    if (*Bmag <= 0.0) {
        bhat[0] = bhat[1] = bhat[2] = 0.0;
        u_E[0] = u_E[1] = u_E[2] = 0.0;
        return;
    }

    const double inv_B = 1.0 / *Bmag;
    bhat[0] = B[0]*inv_B; bhat[1] = B[1]*inv_B; bhat[2] = B[2]*inv_B;

    /* With E = -(Omega x r) x B the E x B drift is exactly the perpendicular part of the
     * corotation velocity, so corotation needs no separate term. */
    double omega_cross_r[3];
    v3_cross(g_lorentz.rotation, r_rel, omega_cross_r);
    const double along = v3_dot(omega_cross_r, bhat);
    u_E[0] = omega_cross_r[0] - along*bhat[0];
    u_E[1] = omega_cross_r[1] - along*bhat[1];
    u_E[2] = omega_cross_r[2] - along*bhat[2];
}

/* Expansion parameter of the drift approximation: gyroradius over field scale length.
 *   r_gyro = v_perp / (|q/m| B) = sqrt(2 mu_bar B) / (|q/m| B)
 *   L      = B / |grad B|      (~ r/3 for a dipole)
 * A dipole falls off fast enough that epsilon grows like r^2, so this is what draws the
 * outer edge of the magnetosphere for a given particle energy. */
static double gc_epsilon(double Bmag, const double gradB[3], double mu_bar, double q_over_m)
{
    if (Bmag <= 0.0 || q_over_m == 0.0) return INFINITY;
    if (mu_bar <= 0.0) return 0.0;
    const double r_gyro = sqrt(2.0 * mu_bar * Bmag) / (fabs(q_over_m) * Bmag);
    return r_gyro * v3_norm(gradB) / Bmag;
}

/* Smooth cut-off on the magnetic transport terms. Particles are classified as magnetised
 * or not once per advance call; this only keeps the right-hand side bounded for a particle
 * that crosses the boundary part way through a call, before it is reclassified. It is a
 * numerical guard rail, not physics: a de-magnetised particle stops being transported and
 * is handed back to ballistic motion on the next call. */
static double gc_magnetisation_weight(double eps)
{
    if (!(eps < INFINITY)) return 0.0;
    const double x = eps / g_gc_eps_max;
    const double x4 = x*x*x*x;
    return 1.0 / (1.0 + x4*x4);
}

/*
 * dR/dt   = V_central + v_par bhat + u_E + v_gradB + v_curv + v_force
 * dv_par/dt = -mu_bar (bhat . grad|B|) + bhat . a_eff
 *
 * In a current-free region the field-line curvature is kappa = grad_perp|B| / |B|, so the
 * grad-B and curvature drifts collapse onto the same bhat x grad|B| direction and can be
 * summed with a single coefficient.
 */
static void gc_derivs(const body_track_t* bt, double t, const double y[4],
                      double mu_bar, double q_over_m, double beta,
                      double* scratch, double dydt[4])
{
    double* bx = scratch;
    double* bv = scratch + (size_t)bt->n_bodies * 3;
    bodies_at(bt, t, bx, bv);

    const double* cx = &bx[bt->central * 3];
    const double* cv = &bv[bt->central * 3];

    const double pos[3]   = { y[0], y[1], y[2] };
    const double r_rel[3] = { pos[0]-cx[0], pos[1]-cx[1], pos[2]-cx[2] };
    const double v_par    = y[3];

    double bhat[3], Bmag, gradB[3], u_E[3];
    gc_local_field(t, r_rel, bhat, &Bmag, gradB, u_E);

    double a_eff[3];
    gc_effective_acceleration(bt, bx, pos, r_rel, beta, a_eff);

    if (Bmag <= 0.0 || q_over_m == 0.0) {
        /* No field to guide along: coast with the central body. */
        dydt[0] = cv[0]; dydt[1] = cv[1]; dydt[2] = cv[2];
        dydt[3] = 0.0;
        return;
    }

    const double v_perp2 = 2.0 * mu_bar * Bmag;
    const double w = gc_magnetisation_weight(gc_epsilon(Bmag, gradB, mu_bar, q_over_m));

    double b_cross_gradB[3];
    v3_cross(bhat, gradB, b_cross_gradB);
    const double drift_coeff = (0.5*v_perp2 + v_par*v_par) / (q_over_m * Bmag * Bmag);

    double a_cross_b[3];
    v3_cross(a_eff, bhat, a_cross_b);
    const double force_coeff = 1.0 / (q_over_m * Bmag);

    for (int i = 0; i < 3; i++) {
        dydt[i] = cv[i] + w * (v_par * bhat[i]
                             + u_E[i]
                             + drift_coeff * b_cross_gradB[i]
                             + force_coeff * a_cross_b[i]);
    }

    dydt[3] = w * (-mu_bar * v3_dot(bhat, gradB) + v3_dot(bhat, a_eff));
}

/* Split a full particle velocity into guiding-centre coordinates.
 *
 * v_par = v_rel . bhat holds both for a freshly picked-up ion and for a guiding centre
 * returning from a previous call, because every drift term is perpendicular to bhat.
 * mu is only derived on the first call after ionisation: whatever perpendicular velocity
 * a neutral has on top of the corotation drift becomes gyration, which is exactly the
 * pickup energy. Afterwards mu travels with the particle as an adiabatic invariant,
 * since the gyration is no longer represented in the state vector. */
static void gc_seed(const body_track_t* bt, gc_ion_t* ion, double* scratch)
{
    double* bx = scratch;
    double* bv = scratch + (size_t)bt->n_bodies * 3;
    bodies_at(bt, bt->t0, bx, bv);

    const double* cx = &bx[bt->central * 3];
    const double* cv = &bv[bt->central * 3];

    const double r_rel[3] = { ion->y[0]-cx[0], ion->y[1]-cx[1], ion->y[2]-cx[2] };
    const double v_rel[3] = { ion->v_full[0]-cv[0], ion->v_full[1]-cv[1], ion->v_full[2]-cv[2] };

    double bhat[3], Bmag, gradB[3], u_E[3];
    gc_local_field(bt->t0, r_rel, bhat, &Bmag, gradB, u_E);

    if (Bmag <= 0.0) {
        ion->y[3] = 0.0;
        if (ion->mu_bar < 0.0) ion->mu_bar = 0.0;
        return;
    }

    ion->y[3] = v3_dot(v_rel, bhat);

    if (ion->mu_bar < 0.0) {
        const double w[3] = { v_rel[0]-u_E[0], v_rel[1]-u_E[1], v_rel[2]-u_E[2] };
        double v_perp2 = v3_dot(w, w) - ion->y[3]*ion->y[3];
        if (v_perp2 < 0.0) v_perp2 = 0.0;
        ion->mu_bar = v_perp2 / (2.0 * Bmag);
    }
}

/* A guiding centre that ends the step inside a gravitating body has hit the surface and
 * is absorbed. Test particles carry no radius of their own, so REBOUND's collision search
 * never removed them either; without a sink here they would keep free-falling on a point
 * mass and pick up arbitrarily large parallel velocities. */
static int gc_absorbed(const body_track_t* bt, const gc_ion_t* ion, double* scratch)
{
    if (bt->radius == NULL) return 0;

    double* bx = scratch;
    double* bv = scratch + (size_t)bt->n_bodies * 3;
    bodies_at(bt, bt->t1, bx, bv);

    for (int j = 0; j < bt->n_bodies; j++) {
        const double rj = bt->radius[j];
        if (rj <= 0.0) continue;
        const double d[3] = { ion->y[0]-bx[j*3], ion->y[1]-bx[j*3+1], ion->y[2]-bx[j*3+2] };
        if (v3_dot(d, d) < rj*rj) return 1;
    }
    return 0;
}

/* Deterministic pseudo-random gyrophase (splitmix-style bit mix of the particle hash).
 * The guiding-centre state deliberately does not carry a gyrophase, so one has to be
 * drawn when gyration is reinstated. Seeding from the hash keeps runs reproducible. */
static double gyrophase_from_hash(uint32_t h)
{
    h ^= h >> 16; h *= 0x7feb352dU;
    h ^= h >> 15; h *= 0x846ca68bU;
    h ^= h >> 16;
    return ((double)h / 4294967296.0) * 2.0 * M_PI;
}

/* Classify a charged particle at the start of an advance call.
 *
 * Returns 1 if it is magnetised enough for the drift approximation. If it is not, and it
 * was being tracked as a guiding centre (mu_bar >= 0), the gyration speed is returned to
 * `v_restore` at a random gyrophase: the drift state carries the pickup energy in mu
 * rather than in the velocity, so dropping straight back to ballistic motion would
 * silently discard it and bias escape rates low.
 */
static int gc_classify(double t, const double central_pos[3], const double central_vel[3],
                       const double pos[3], const double vel[3],
                       double mu_bar, double q_over_m, uint32_t hash,
                       double v_restore[3])
{
    v_restore[0] = vel[0]; v_restore[1] = vel[1]; v_restore[2] = vel[2];
    if (q_over_m == 0.0) return 0;

    const double r_rel[3] = { pos[0]-central_pos[0], pos[1]-central_pos[1], pos[2]-central_pos[2] };
    const double v_rel[3] = { vel[0]-central_vel[0], vel[1]-central_vel[1], vel[2]-central_vel[2] };

    double bhat[3], Bmag, gradB[3], u_E[3];
    gc_local_field(t, r_rel, bhat, &Bmag, gradB, u_E);
    if (Bmag <= 0.0) return 0;

    /* An unseeded particle has no mu yet, so take the gyration speed it would be given. */
    double mu_probe = mu_bar;
    if (mu_probe < 0.0) {
        const double v_par = v3_dot(v_rel, bhat);
        const double w[3] = { v_rel[0]-u_E[0], v_rel[1]-u_E[1], v_rel[2]-u_E[2] };
        double v_perp2 = v3_dot(w, w) - v_par*v_par;
        if (v_perp2 < 0.0) v_perp2 = 0.0;
        mu_probe = v_perp2 / (2.0 * Bmag);
    }

    if (gc_epsilon(Bmag, gradB, mu_probe, q_over_m) < g_gc_eps_max) return 1;

    if (mu_bar >= 0.0) {
        /* Rebuild a gyration velocity of the right magnitude perpendicular to B. */
        double e1[3], e2[3];
        v3_perpendicular(bhat, e1);
        v3_cross(bhat, e1, e2);
        const double v_perp = sqrt(2.0 * mu_bar * Bmag);
        const double phase = gyrophase_from_hash(hash);
        for (int i = 0; i < 3; i++) {
            v_restore[i] = vel[i] + v_perp * (cos(phase)*e1[i] + sin(phase)*e2[i]);
        }
    }
    return 0;
}

/* ========================================================================
 *  Embedded Runge-Kutta (Cash-Karp 4/5) for the guiding-centre state
 * ======================================================================== */

static void gc_rk_step(const body_track_t* bt, double t, const double y[4],
                       const double dydt[4], double h, const gc_ion_t* ion,
                       double* scratch, double yout[4], double yerr[4])
{
    static const double
        a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875,
        b21 = 0.2,
        b31 = 3.0/40.0,       b32 = 9.0/40.0,
        b41 = 0.3,            b42 = -0.9,          b43 = 1.2,
        b51 = -11.0/54.0,     b52 = 2.5,           b53 = -70.0/27.0,   b54 = 35.0/27.0,
        b61 = 1631.0/55296.0, b62 = 175.0/512.0,   b63 = 575.0/13824.0,
        b64 = 44275.0/110592.0, b65 = 253.0/4096.0,
        c1 = 37.0/378.0,      c3 = 250.0/621.0,    c4 = 125.0/594.0,   c6 = 512.0/1771.0,
        dc1 = 37.0/378.0    - 2825.0/27648.0,
        dc3 = 250.0/621.0   - 18575.0/48384.0,
        dc4 = 125.0/594.0   - 13525.0/55296.0,
        dc5 = -277.0/14336.0,
        dc6 = 512.0/1771.0  - 0.25;
    (void)force;
    if (!g_lorentz.enabled) return;

    const double mu = ion->mu_bar, qm = ion->q_over_m, beta = ion->beta;
    double ytemp[4], k2[4], k3[4], k4[4], k5[4], k6[4];

    for (int i = 0; i < 4; i++) ytemp[i] = y[i] + h*b21*dydt[i];
    gc_derivs(bt, t + a2*h, ytemp, mu, qm, beta, scratch, k2);

    for (int i = 0; i < 4; i++) ytemp[i] = y[i] + h*(b31*dydt[i] + b32*k2[i]);
    gc_derivs(bt, t + a3*h, ytemp, mu, qm, beta, scratch, k3);

    for (int i = 0; i < 4; i++) ytemp[i] = y[i] + h*(b41*dydt[i] + b42*k2[i] + b43*k3[i]);
    gc_derivs(bt, t + a4*h, ytemp, mu, qm, beta, scratch, k4);

    for (int i = 0; i < 4; i++)
        ytemp[i] = y[i] + h*(b51*dydt[i] + b52*k2[i] + b53*k3[i] + b54*k4[i]);
    gc_derivs(bt, t + a5*h, ytemp, mu, qm, beta, scratch, k5);

    for (int i = 0; i < 4; i++)
        ytemp[i] = y[i] + h*(b61*dydt[i] + b62*k2[i] + b63*k3[i] + b64*k4[i] + b65*k5[i]);
    gc_derivs(bt, t + a6*h, ytemp, mu, qm, beta, scratch, k6);

    for (int i = 0; i < 4; i++) {
        yout[i] = y[i] + h*(c1*dydt[i] + c3*k3[i] + c4*k4[i] + c6*k6[i]);
        yerr[i] = h*(dc1*dydt[i] + dc3*k3[i] + dc4*k4[i] + dc5*k5[i] + dc6*k6[i]);
    }
}

/* Advance one guiding centre across the whole REBOUND step with adaptive substepping.
 *
 * Unlike IAS15 this controller carries no state between calls and does not need to: the
 * drift timescale is comparable to the advance interval itself, so the very first trial
 * step is already close to the accepted one. */
static void gc_advance_ion(const body_track_t* bt, gc_ion_t* ion, double* scratch, double rtol)
{
    const double span = bt->t1 - bt->t0;
    if (span == 0.0) return;

    double t = bt->t0;
    double h = span;
    const double h_min = fabs(span) * 1e-10;
    double y[4];
    memcpy(y, ion->y, sizeof(y));

    for (int step = 0; step < GC_MAX_STEPS; step++) {
        if ((span > 0.0 && t >= bt->t1) || (span < 0.0 && t <= bt->t1)) break;
        if ((span > 0.0 && t + h > bt->t1) || (span < 0.0 && t + h < bt->t1)) h = bt->t1 - t;

        double dydt[4], ytemp[4], yerr[4];
        gc_derivs(bt, t, y, ion->mu_bar, ion->q_over_m, ion->beta, scratch, dydt);
        gc_rk_step(bt, t, y, dydt, h, ion, scratch, ytemp, yerr);

        double errmax = 0.0;
        for (int i = 0; i < 4; i++) {
            const double yscal = fabs(y[i]) + fabs(h*dydt[i]) + 1e-30;
            const double e = fabs(yerr[i] / yscal);
            if (e > errmax) errmax = e;
        }
        errmax /= rtol;

        if (errmax > 1.0 && fabs(h) > h_min) {
            double shrink = 0.9 * pow(errmax, -0.25);
            if (shrink < 0.1) shrink = 0.1;
            h *= shrink;
            continue;
        }

        t += h;
        memcpy(y, ytemp, sizeof(y));

        double grow = (errmax > 1.89e-4) ? 0.9 * pow(errmax, -0.2) : 5.0;
        if (grow > 5.0) grow = 5.0;
        h *= grow;
    }

    memcpy(ion->y, y, sizeof(y));

    /* The guiding-centre velocity is dR/dt itself, so report the right-hand side at the
     * end of the step back to REBOUND. */
    double dydt_end[4];
    gc_derivs(bt, bt->t1, ion->y, ion->mu_bar, ion->q_over_m, ion->beta, scratch, dydt_end);
    ion->v_full[0] = dydt_end[0];
    ion->v_full[1] = dydt_end[1];
    ion->v_full[2] = dydt_end[2];
}

/* ========================================================================
 *  Thread worker for parallel integration
 * ======================================================================== */

typedef struct {
    struct reb_simulation*  sim;
    struct rebx_extras*     rebx;
    double                  t0;
    double                  target_time;
    double                  max_dt;
    int                     status;
    int                     started;

    /* Charged test particles, held outside the REBOUND simulation */
    gc_ion_t*               ions;
    int                     n_ions;
    double                  gc_rtol;

    /* Per-thread buffers for the guiding-centre pusher */
    body_track_t            bt;
    double*                 body_mass;
    int*                    body_rad_source;
    double*                 body_x0;
    double*                 body_v0;
    double*                 body_x1;
    double*                 body_v1;
    double*                 scratch;
} worker_arg_t;

/* Grain impacts remove only the tracer; other pairs retain historical merging. */
static enum REB_COLLISION_RESOLVE_OUTCOME resolve_collision(struct reb_simulation* sim, struct reb_collision collision)
{
    struct rebx_extras* rebx = sim->extras;
    int* kind1 = rebx_get_param(rebx, sim->particles[collision.p1].ap, "particle_kind");
    int* kind2 = rebx_get_param(rebx, sim->particles[collision.p2].ap, "particle_kind");
    if (collision.p1 >= sim->N_active && kind1 && *kind1 == 1 && collision.p2 < sim->N_active)
        return 1;
    if (collision.p2 >= sim->N_active && kind2 && *kind2 == 1 && collision.p1 < sim->N_active)
        return 2;
    if ((kind1 && *kind1 == 1) || (kind2 && *kind2 == 1)) return 0;
    return reb_collision_resolve_merge(sim, collision);
}

static void bodies_snapshot(struct reb_simulation* sim, int n_bodies, double* x, double* v)
{
    for (int i = 0; i < n_bodies; i++) {
        struct reb_particle* p = &sim->particles[i];
        x[i*3 + 0] = p->x;  x[i*3 + 1] = p->y;  x[i*3 + 2] = p->z;
        v[i*3 + 0] = p->vx; v[i*3 + 1] = p->vy; v[i*3 + 2] = p->vz;
    }
}

static void* worker_thread(void* arg)
{
    worker_arg_t* w = (worker_arg_t*)arg;

    while (w->sim->t < w->target_time) {
        double end = w->target_time;
        if (w->max_dt > 0.0) end = fmin(end, w->sim->t + w->max_dt);
        if (end <= w->sim->t) {
            w->status = REB_STATUS_GENERIC_ERROR;
            break;
        }
        w->status = reb_simulation_integrate(w->sim, end);
        if (w->status != REB_STATUS_SUCCESS) break;
    }

    const int n_bodies = w->bt.n_bodies;

    /* Bracket the REBOUND step with snapshots of the gravitating bodies so the
     * guiding-centre pusher can interpolate them onto its own substeps. */
    if (n_bodies > 0) {
        bodies_snapshot(w->sim, n_bodies, w->body_x0, w->body_v0);
    }

    reb_simulation_integrate(w->sim, w->target_time);

    if (n_bodies > 0) {
        /* A merge among the active bodies would invalidate the pairing; in that case
         * fall back to holding them fixed at their pre-step state. */
        if ((int)w->sim->N_active >= n_bodies) {
            bodies_snapshot(w->sim, n_bodies, w->body_x1, w->body_v1);
        } else {
            memcpy(w->body_x1, w->body_x0, (size_t)n_bodies * 3 * sizeof(double));
            memcpy(w->body_v1, w->body_v0, (size_t)n_bodies * 3 * sizeof(double));
        }
    }

    w->bt.t0 = w->t0;
    w->bt.t1 = w->sim->t;

    for (int i = 0; i < w->n_ions; i++) {
        gc_seed(&w->bt, &w->ions[i], w->scratch);
        gc_advance_ion(&w->bt, &w->ions[i], w->scratch, w->gc_rtol);
        w->ions[i].absorbed = gc_absorbed(&w->bt, &w->ions[i], w->scratch);
    }

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
 *    mu_values       – double array [n_total]  (mu/m values, <0 = derive from velocity)
 *    radii           – double array [n_total]  (absorbing radii; only the active ones are
 *                      used, as a sink for guiding centres that reach a body's surface)
 *    rad_source_flags– int array [n_total]     (1 if radiation source, 0 otherwise)
 *    target_time     – integration target time (absolute)
 *    G_value         – gravitational constant
 *    min_dt          – minimum timestep for IAS15
 *    gc_rtol         – relative tolerance of the guiding-centre pusher
 *    gc_eps_max      – gyroradius/scale-length above which a particle is treated as
 *                      unmagnetised and handed back to ballistic integration
 *    n_threads       – number of threads to use
 *    max_dt          – optional maximum collision-search interval (0 disables)
 *    state_out       – output array, same layout as state_in (preallocated)
 *    hashes_out      – output: particle hashes in output order
 *    qm_out          – output: q/m values in output order
 *    mu_out          – output: mu/m values in output order
 *    n_out           – output: actual number of particles after merges/collisions
 *    sim_time_out    – output: final simulation time
 * ======================================================================== */

int serpens_advance_integrate(
    int n_active,
    int n_total,
    const double* state_in,
    const uint32_t* hashes_in,
    const double* beta_values,
    const double* qm_values,
    const double* mu_values,
    const double* radii,
    const int* rad_source_flags,
    const uint32_t* source_primary_hashes,
    const double* radii_in,
    const int* particle_kinds,
    double target_time,
    double G_value,
    double min_dt,
    double gc_rtol,
    double gc_eps_max,
    double epsilon,
    double sim_t0,
    double initial_dt,
    double max_dt,
    double radiation_c,
    int n_threads,
    int force_is_velocity_dependent,
    /* outputs */
    double* state_out,
    uint32_t* hashes_out,
    double* qm_out,
    double* mu_out,
    double* radii_out,
    int* n_out,
    int* n_active_out,
    double* sim_time_out)
{
    int has_grains = 0;
    for (int i = n_active; i < n_total; i++) {
        if (particle_kinds[i] == 1) has_grains = 1;
    }
    if (gc_eps_max > 0.0) g_gc_eps_max = gc_eps_max;

    if (n_threads < 1) n_threads = 1;
    int n_test = n_total - n_active;
    if (n_threads > n_test && n_test > 0) n_threads = n_test;
    if (n_test == 0) n_threads = 1;

    /* --- Allocate worker data --- */
    worker_arg_t* workers = (worker_arg_t*)calloc(n_threads, sizeof(worker_arg_t));
    pthread_t*    threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
    if (!workers || !threads) {
        free(workers);
        free(threads);
        return REB_STATUS_GENERIC_ERROR;
    }

    /* The gravitating bodies are identical in every thread, so a single mass /
     * radiation-source table is shared by all of them. */
    double* body_mass       = (double*)calloc((size_t)(n_active > 0 ? n_active : 1), sizeof(double));
    int*    body_rad_source = (int*)calloc((size_t)(n_active > 0 ? n_active : 1), sizeof(int));
    for (int i = 0; i < n_active; i++) {
        body_mass[i]       = state_in[i*7 + 0];
        body_rad_source[i] = rad_source_flags[i];
    }

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
        sim->ri_ias15.epsilon = epsilon;
        sim->collision = has_grains ? REB_COLLISION_LINE : REB_COLLISION_DIRECT;
        sim->collision_resolve = resolve_collision;
        sim->collision_resolve_keep_sorted = 1;
        sim->exact_finish_time = 1;
        sim->dt = initial_dt;
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
            p.r = radii_in[i];
            reb_simulation_add(sim, p);
        }

        /* Split this thread's slice: neutrals go to REBOUND, charged particles are
         * handed to the guiding-centre pusher, which needs equations of motion REBOUND
         * cannot express. With the Lorentz force disabled everything stays in REBOUND. */
        gc_ion_t* ions = (gc_ion_t*)calloc((size_t)(my_test_count > 0 ? my_test_count : 1),
                                           sizeof(gc_ion_t));
        int n_ions = 0;

        /* Global index of each neutral, in the order REBOUND holds them */
        int* neutral_gi = (int*)calloc((size_t)(my_test_count > 0 ? my_test_count : 1),
                                       sizeof(int));
        int n_neutral = 0;

        const int ci = (g_lorentz.central_index >= 0 && g_lorentz.central_index < n_active)
                       ? g_lorentz.central_index : 0;
        const double* central_pos = (n_active > 0) ? &state_in[ci*7 + 1] : NULL;
        const double* central_vel = (n_active > 0) ? &state_in[ci*7 + 4] : NULL;

        for (int j = 0; j < my_test_count; j++) {
            int gi = n_active + test_offset + j;  /* global index */

            double vel[3] = { state_in[gi*7 + 4], state_in[gi*7 + 5], state_in[gi*7 + 6] };

            if (g_lorentz.enabled && qm_values[gi] != 0.0 && central_pos != NULL) {
                const double pos[3] = { state_in[gi*7 + 1], state_in[gi*7 + 2], state_in[gi*7 + 3] };
                double v_restore[3];
                int magnetised = gc_classify(sim_t0, central_pos, central_vel, pos, vel,
                                             mu_values[gi], qm_values[gi], hashes_in[gi],
                                             v_restore);
                if (magnetised) {
                    gc_ion_t* ion = &ions[n_ions++];
                    ion->y[0] = pos[0];
                    ion->y[1] = pos[1];
                    ion->y[2] = pos[2];
                    ion->y[3] = 0.0;
                    ion->v_full[0] = vel[0];
                    ion->v_full[1] = vel[1];
                    ion->v_full[2] = vel[2];
                    ion->mass     = state_in[gi*7 + 0];
                    ion->mu_bar   = mu_values[gi];
                    ion->q_over_m = qm_values[gi];
                    ion->beta     = beta_values[gi];
                    ion->hash     = hashes_in[gi];
                    continue;
                }
                /* Too weakly magnetised for the drift approximation: fall back to
                 * ballistic motion in REBOUND with the gyration energy restored. */
                vel[0] = v_restore[0]; vel[1] = v_restore[1]; vel[2] = v_restore[2];
            }

            struct reb_particle p = {0};
            p.m  = state_in[gi*7 + 0];
            p.x  = state_in[gi*7 + 1];
            p.y  = state_in[gi*7 + 2];
            p.z  = state_in[gi*7 + 3];
            p.vx = vel[0];
            p.vy = vel[1];
            p.vz = vel[2];
            p.hash = hashes_in[gi];
            p.r = radii_in[gi];
            reb_simulation_add(sim, p);
            neutral_gi[n_neutral++] = gi;
        }

        sim->N_active = n_active;

        /* Attach REBOUNDx and set per-particle parameters */
        struct rebx_extras* rebx = rebx_attach(sim);

        /* Register custom parameters */
        rebx_register_param(rebx, "q_over_m", REBX_TYPE_DOUBLE);
        rebx_register_param(rebx, "source_primary", REBX_TYPE_UINT32);
        rebx_register_param(rebx, "particle_kind", REBX_TYPE_INT);

        /* Radiation forces */
        struct rebx_force* rf = rebx_load_force(rebx, "radiation_forces");
        rebx_add_force(rebx, rf);
        rebx_set_param_double(rebx, &rf->ap, "c", radiation_c);
        if (g_lorentz.enabled) {
            struct rebx_force* lf = rebx_create_force(rebx, "lorentz_force");
            lf->force_type = REBX_FORCE_VEL;
            lf->update_accelerations = lorentz_force;
            rebx_add_force(rebx, lf);
        }
        sim->force_is_velocity_dependent |= force_is_velocity_dependent;

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
            /* Preserve source identity without modifying its orbit in workers. */
            if (source_primary_hashes[i] != 0) {
                rebx_set_param_uint32(rebx, (struct rebx_node**)&sim->particles[i].ap, "source_primary", source_primary_hashes[i]);
            }
        }

        /* Test particles left in REBOUND: set beta and q_over_m */
        for (int j = 0; j < n_neutral; j++) {
            int gi = neutral_gi[j];
            int li = n_active + j;  /* local index in this sim */
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "beta", beta_values[gi]);
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "q_over_m", qm_values[gi]);
            rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[li].ap, "particle_kind", particle_kinds[gi]);
            if (rad_source_flags[gi]) {
                rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[li].ap, "radiation_source", 1);
            }
        }
        free(neutral_gi);

        workers[t].sim          = sim;
        workers[t].rebx         = rebx;
        workers[t].t0           = sim_t0;
        workers[t].target_time  = target_time;
        workers[t].max_dt       = max_dt;
        workers[t].fix_circular = fix_circular;
        workers[t].ions         = ions;
        workers[t].n_ions       = n_ions;
        workers[t].gc_rtol      = gc_rtol > 0.0 ? gc_rtol : 1e-6;

        const size_t body_stride = (size_t)(n_active > 0 ? n_active : 1) * 3;
        workers[t].body_x0 = (double*)calloc(body_stride, sizeof(double));
        workers[t].body_v0 = (double*)calloc(body_stride, sizeof(double));
        workers[t].body_x1 = (double*)calloc(body_stride, sizeof(double));
        workers[t].body_v1 = (double*)calloc(body_stride, sizeof(double));
        workers[t].scratch = (double*)calloc(body_stride * 2, sizeof(double));

        workers[t].bt.n_bodies   = n_active;
        workers[t].bt.central    = (g_lorentz.central_index >= 0 && g_lorentz.central_index < n_active)
                                   ? g_lorentz.central_index : 0;
        workers[t].bt.G          = G_value;
        workers[t].bt.t0         = sim_t0;
        workers[t].bt.t1         = target_time;
        workers[t].bt.mass       = body_mass;
        workers[t].bt.radius     = radii;
        workers[t].bt.rad_source = body_rad_source;
        workers[t].bt.x0         = workers[t].body_x0;
        workers[t].bt.v0         = workers[t].body_v0;
        workers[t].bt.x1         = workers[t].body_x1;
        workers[t].bt.v1         = workers[t].body_v1;

        test_offset += my_test_count;
    }

    /* --- Launch threads --- */
    for (int t = 0; t < n_threads; t++) {
        workers[t].started = pthread_create(&threads[t], NULL, worker_thread, &workers[t]) == 0;
        if (!workers[t].started) worker_thread(&workers[t]);
    }

    /* --- Join threads --- */
    int status = REB_STATUS_SUCCESS;
    for (int t = 0; t < n_threads; t++) {
        if (workers[t].started) pthread_join(threads[t], NULL);
        if (workers[t].status != REB_STATUS_SUCCESS) status = workers[t].status;
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
        qm_out[out_idx] = 0.0;
        mu_out[out_idx] = -1.0;
        radii_out[out_idx] = p->r;
        out_idx++;
    }

    /* Test particles from each thread: REBOUND-integrated neutrals first, then the
     * guiding centres. q/m and mu/m travel in the output ordering so that a collision
     * or merge cannot shift them onto the wrong particle. */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        struct rebx_extras* rebx = workers[t].rebx;

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

            double* qm_ptr = rebx_get_param(rebx, p->ap, "q_over_m");
            qm_out[out_idx] = (qm_ptr != NULL) ? *qm_ptr : 0.0;
            mu_out[out_idx] = -1.0;
            radii_out[out_idx] = p->r;
            out_idx++;
        }

        for (int i = 0; i < workers[t].n_ions; i++) {
            gc_ion_t* ion = &workers[t].ions[i];
            if (ion->absorbed) continue;
            state_out[out_idx*7 + 0] = ion->mass;
            state_out[out_idx*7 + 1] = ion->y[0];
            state_out[out_idx*7 + 2] = ion->y[1];
            state_out[out_idx*7 + 3] = ion->y[2];
            state_out[out_idx*7 + 4] = ion->v_full[0];
            state_out[out_idx*7 + 5] = ion->v_full[1];
            state_out[out_idx*7 + 6] = ion->v_full[2];
            hashes_out[out_idx] = ion->hash;
            qm_out[out_idx] = ion->q_over_m;
            mu_out[out_idx] = ion->mu_bar;
            out_idx++;
        }
    }

    *n_out = out_idx;
    *n_active_out = workers[0].sim->N_active;
    *sim_time_out = workers[0].sim->t;

    /* --- Cleanup --- */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        struct rebx_extras* rebx = workers[t].rebx;

        if (sim) {
            reb_simulation_free(sim);
            workers[t].sim = NULL;
        }

        free(workers[t].ions);
        free(workers[t].body_x0);
        free(workers[t].body_v0);
        free(workers[t].body_x1);
        free(workers[t].body_v1);
        free(workers[t].scratch);
        if (rebx) rebx_free(rebx);
    }
    free(body_mass);
    free(body_rad_source);
    free(workers);
    free(threads);
    return status;
}
