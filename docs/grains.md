# Grain test-particle baseline

Grains are a per-tracer, spherical-grain baseline, not a calibrated silicate or
ice model. Each tracer carries a physical radius, mass, radiation-pressure ratio,
prescribed charge-to-mass ratio, and immutable physical number weight. Gravity
and radiation determine trajectories; this alone does not establish material
survival or a long-lived dust torus.

## Installation

For an interactive walkthrough, open [the grain notebook](../notebooks/grains.ipynb).
It demonstrates population configuration, isolated simulation runs, archive
budgets, size-selected fields, and a tracer-count comparison. Run its cells in
order using a kernel with the project requirements and `ipykernel` installed.

The notebook defaults to a **55-Cnc-e-like ejecting planet**.
Planet masses, radii, and orbit match the `55-Cnc-e` entry in
`resources/objects.json`; the luminosity and fast launch law are illustrative
inputs, not a calibrated dust-production or survival model.

Follow the [normal installation](../README.md#installation), including the normal
Python requirements in `requirements.txt`. For the C backend, build the pinned
REBOUND/REBOUNDx libraries described there and run `make build` from the project
root. The grain bridge changes the C ABI: rebuild existing binaries with
`make -B build` rather than reusing an older shared library.

## Explicit population configuration

Import `GrainPopulation` from `src.grains`. All dimensional inputs use SI units.
**Every field in the following table is currently required**, even when a
particular experiment uses zero luminosity or a single grain size. There are no
implicit material or stellar defaults.

| Required field | Meaning |
| --- | --- |
| `name` | Nonblank descriptive population name. |
| `density_kg_m3` | Positive homogeneous material density, kg/m³. |
| `radius_min_m`, `radius_max_m` | Positive, ordered grain-radius bounds, m; equal bounds give a monodisperse population. |
| `size_slope` | Number-distribution exponent: `dN/da ∝ a^(-size_slope)`, not a mass distribution. |
| `tracers_per_injection` | Positive integer number of numerical tracers per injection. |
| `dust_mass_per_sec` | Nonnegative physical dust production rate, kg/s. |
| `speed_ref_m_s` | Nonnegative launch speed at the reference radius, m/s. |
| `radius_ref_m` | Positive reference grain radius, m. |
| `speed_exponent` | Exponent in the size-dependent launch law below. |
| `launch_altitude_m` | Nonnegative height above the emitting body's surface, m. |
| `radiation_source` | Explicit name of an existing, massive active body. |
| `luminosity_w` | Nonnegative prescribed source luminosity, W. |
| `q_pr` | Nonnegative dimensionless constant, or a table of `(radius_m, q_pr)` pairs. |

Optional fields are `potential_v=0.0` (prescribed potential in volts),
`speed_min_m_s=None`, and `speed_max_m_s=None`. Supplied speed bounds must be
nonnegative and ordered. All numerical inputs must be finite.

The launch-speed magnitude is
`v(a) = speed_ref_m_s * (a / radius_ref_m)^(-speed_exponent)`, clipped to any
supplied speed bounds. A positive exponent makes larger grains slower.
`q_pr` tables need at least two strictly increasing positive radius entries and
must cover the entire population radius interval. Interpolation is linear in
radius; extrapolation is rejected. `q_pr` is a prescribed radiation-pressure
efficiency, not an absorption, emission, or extinction model.

Register an existing emitting body using
`simulation.object_to_source(name, species=None, *, grains=None)`. Pass a single
`GrainPopulation` or an iterable as `grains`; gas `species` can coexist with
grains, but at least one is required. For example, after constructing a population
and the system, use `simulation.object_to_source("moon", grains=population)`.
Grain population IDs are separate from chemical species IDs. Grain tracers are
explicitly tagged with `particle_kind=1`, do not undergo chemical reactions,
and are not subject to gas lifetime/decay weighting.

### Ejecting planets without moons

The source can be a planet directly orbiting the radiation-source star; no dummy
moon or alternate grain implementation is needed. After adding the star and
planet (with `primary='star'`), use:

```python
simulation.object_to_source('planet', grains=population)
simulation.advance_single(time=300., orbit_object='planet')
analyzer = SerpensAnalyzer(reference_system='planet')
```

Keep `population.radiation_source='star'`. Here the numerical removal boundary is
`r_max` times the planet's orbital semimajor axis, measured from the **star**.
For circumstellar radial profiles, pass the star's snapshot coordinates as
`center`. `Visualize(..., reference_system='planet')` likewise centers both planar
and LOS views on the star, and its `lim` uses stellar radii. With a moon source,
the corresponding center and radius unit are those of the planet.


## Physical weights and forces

For physical grain radius `a` and material density `rho`:

- Single-grain mass: `m = (4π/3) rho a³`.
- Radiation/gravity ratio: `beta = 3 L Q_pr / (16π c G M rho a)`, using the
  selected radiation source's mass `M` and luminosity `L`.
- Isolated-sphere charge: `q = 4π epsilon_0 a V`, hence
  `q/m = 3 epsilon_0 V / (rho a²)`; the sign follows the prescribed potential.

These physical properties live in metadata. REBOUND particle `m=0` and `r=0`
keep grains as non-gravitating point test particles, not massive resolved dust
spheres. They can be removed on impact with finite-radius active bodies; they
do not collide, merge, or gravitationally interact with one another.

For an injection interval `dt`, draw radii from the number distribution and give
every tracer in that batch the same number weight:

`W = dust_mass_per_sec * dt / sum(m_i)`.

Thus `sum(W * m_i) = dust_mass_per_sec * dt` for the realized batch. The number,
mass, and geometric cross-section represented by tracer `i` are respectively
`W_i`, `W_i m_i`, and `W_i π a_i²`. Increasing the tracer count changes sampling
resolution, not the prescribed injected mass. 

For neutral Python SERPENS runs, explicitly set
`GLOBAL_PARAMETERS.set("lorentz_enabled", False)` and use `potential_v=0`.
Nonzero potential requires CERPENS and `lorentz_enabled=True`; unsupported
charged configurations are rejected rather than silently integrated as neutral.
CERPENS uses its existing prescribed magnetic-field model. This is not a charging
solver, and the field geometry, strength, and rotation assumptions must suit the
system being studied.


## Archives and loss accounting

`simdata/particle_params.h5` uses HDF5 `metadata_version=1`. It stores population
configurations under `grain_populations`,
and hash-keyed per-tracer `particle_kind`, `grain_population_id`,
`grain_radius_m`, `grain_mass_kg`, `grain_number_weight`, `beta`, `q_over_m`,
`source_hash`, and `serpens_creation_time`. Keep this metadata with the simulation
snapshots: REBOUND's zero particle masses/radii cannot reconstruct physical
grain budgets. Legacy gas archives remain distinct from malformed grain records.


## Analysis and plotting

Use grain-specific methods on a `SerpensAnalyzer` instance. `timestep` selects an
archived snapshot, not a duration in seconds. Grain diagnostics ignore gas decay.

- `analyzer.grain_field(timestep, quantity='number', d=3, population_id=None, source_hash=None, radius_range=None)`
  returns a dictionary with `positions`, `density`, `weights`, `hashes`, `units`,
  `estimator`, `dimension`, and `quantity`. `d=2` projects onto the xy plane;
  `d=3` uses xyz. Positions are metres, weights are physical totals, and density
  is a DTFE vertex estimate. Population/source filters accept IDs/hashes;
  `radius_range=(minimum, maximum)` is an inclusive **grain-size** interval in
  metres, not a spatial distance. Analyzer cutoffs apply.
- `analyzer.grain_radial_profile(timestep, bins, quantity='number', center=None, population_id=None, source_hash=None, radius_range=None)`
  bins physical weights by cylindrical xy distance. `bins` is a positive count
  or strictly increasing nonnegative radius edges in metres. The default center
  is the SI origin; pass the desired body's snapshot position for a body-centered
  profile (a third coordinate is ignored). The dictionary contains `bin_edges`,
  bin-center `radii`, `integrated_weights`, `column_density`, `annulus_areas`,
  `center`, `quantity`, `units`, `weight_units`, and `radius_units`. Column density
  is integrated weight divided by annulus area in m², not a DTFE estimate.
  Analyzer cutoffs and the same grain filters apply.
- `analyzer.grain_budget(timestep, population_id=None, source_hash=None, radius_range=None)`
  returns `injected`, `retained`, `removed`, `removed_by_cause`, `unaccounted`,
  and `closure_residual`; each total (and each cause) has `number`, `mass`, and
  `area` entries. It also returns `unaccounted_hashes`, `time`, and `units`.
  `closure_residual = injected - retained - removed`; missing tracers without an
  effective recorded loss remain explicitly unaccounted, not silently counted
  as escape. Budgets use the full snapshot/history and ignore display cutoffs.

| `quantity` | Weight units | `d=2` density units | `d=3` density units |
| --- | --- | --- | --- |
| `'number'` | 1 (physical grains) | m⁻² | m⁻³ |
| `'mass'` | kg | kg m⁻² | kg m⁻³ |
| `'area'` | m² (geometric cross-section) | 1 | m⁻¹ |

Import `Visualize` from `src.visualizing.visualize`, construct
`Visualize(rebsim, reference_system, panel_count=1)`, and call
`visualizer.add_grain_field(field)` with the returned field dictionary. Grain
plots retain SI quantity labels instead of gas cm-based density conversions.
Geometric cross-section, even its dimensionless projected density, is **not
optical depth**.


## Omitted Physics

Excluded physics includes sublimation/sputtering erosion, fragmentation,
coagulation, grain–grain collisions, evolving charge, plasma/gas drag, shadowing,
radiative transfer, wavelength-dependent optical material modeling, dust–gas
feedback, and dust self-gravity. PR drag is included. 