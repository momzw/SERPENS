# Interactive visualization

The local dashboard and notebook figures share scientific data preparation and
Plotly builders. Existing `SerpensAnalyzer` APIs (`plot_planar`,
`plot_lineofsight`, `plot_1d_cut`, `plot_3d`, `calculate_phasecurve`, and
`plot_phasecurve`) remain available; this interface does not replace them.

## Install and launch

Python >=3.9 is required. From the SERPENS repository, install the optional web
extra, which supplies `dash>=2.11,<4` with native Jupyter support:

```bash
python -m pip install -e '.[web]'
```

Dash is loaded lazily when the web interface is needed. Ordinary analyzer and
shared provider/figure use does not require Dash or a separate JupyterDash package.

Change to the project directory and **trusted dataset directory containing existing `simdata/`**
and run:

```bash
python -m src.visualizing.webapp --reference-system planet --port 8050
```

Visit `http://127.0.0.1:8050`. `planet` must identify the intended reference body
in that dataset. Analyzer parameter archives contain pickled objects: load only trusted datasets.
Keep the server bound to localhost; this is not an authenticated public service.


## Jupyter: reuse an existing analyzer

With an existing `SerpensAnalyzer` instance named `analyzer`, the recommended
entry point is:

```python
from src.visualizing.webapp import launch_notebook

app = launch_notebook(analyzer, port=8050)
```

The default is `jupyter_mode='external'`, displaying a link to the local app.
For an embedded view, pass `jupyter_mode='inline'` instead. The helper caches the
app on the analyzer and reuses its server on the same port when a cell is run
again. Asking that analyzer to launch on another port is rejected to prevent
duplicate servers. Choose the desired port on the first launch; restart the
kernel to start a fresh session if necessary.

For manual lifecycle management, the lower-level alternative is:

```python
from src.visualizing.webapp import create_app

app = create_app(analyzer)
app.run(
    host='127.0.0.1', port=8050, debug=False,
    use_reloader=False, jupyter_mode='external',
)
```

Do not run both alternatives. Prefer `launch_notebook` for repeatable notebook
cells rather than repeatedly constructing apps and starting servers yourself.

## Shared provider without a web server

```python
from src.visualizing.data import DisplayOptions, PlotRequest, VisualizationDataProvider
from src.visualizing.figures import build_figure

provider = VisualizationDataProvider(analyzer)
request = PlotRequest(timestep=1, view='planar', species_ids=None, dimension=3)
data = provider.prepare(request)
fig = build_figure(data, DisplayOptions(thresholds={'1': (2, None)}))
fig.show()
```

Choose an existing archive timestep (`1` requires at least two snapshots).
`species_ids=None` selects all gas species; explicit IDs are dataset species
identifiers, not panel indices. The threshold key `'1'` refers to species ID 1.
Builders return figures without displaying or saving them; use `.show()` explicitly.

## Views and scientific controls

| View (`PlotRequest.view`) | Meaning and controls |
| --- | --- |
| Planar (`'planar'`) | Orbital-plane projection, with 2D or 3D density estimation. Select timestep and species; inspect scatter, triangulation, bodies, orbits, shadow and star connection where applicable. |
| Line of sight (`'los'`) | Observer-plane projection. Gas uses the LOS column-density estimator and visibility mask; grain LOS has the different interpretation below. |
| 3D (`'3d'`) | Interactive spatial distribution; requires `dimension=3`. Rotate, zoom, and select whether to show the star. |
| Profile (`'profile'`) | Gas 1D cuts with source selection and distance in source radii (`max_distance_rp`), with raw/Gaussian-smoothed curves and optional logarithmic axes. |
| Phase curve (`'phase'`) | Gas phase statistics: maximum or mean, column density and/or particle density. Explicitly compute from the archive or read an existing local CSV. |

Density views support gas, grain, or mixed fields (`kind='gas'`, `'grain'`, or
`'mixed'`). Grain fields allow population, source, radius-range, and quantity
selection. Grain profile cuts and phase curves are not supported.

### Phase curves and downloads

Selecting a phase view does not silently start a phase calculation. Use the
explicit compute action with one gas species, a valid source, and the desired
orbit span, or load an existing local phase CSV. At provider level, submitting
a phase request without `phase_file` is the explicit compute operation; supplying
`phase_file` reads that file instead. The required numeric, finite CSV columns are
`Phase`, `Timestep`, `Max_2D`, `Mean_2D`, `Max_3D`, and `Mean_3D`.

The shared phase workflow returns data in memory and does not implicitly write
`phase-curve.csv` or overwrite local results. Use the dashboard's CSV download to
save data explicitly and its HTML download to save an interactive figure. The
Plotly modebar's image button downloads an image in the browser; these downloads
do not require Kaleido. Legacy `calculate_phasecurve` retains its existing
file-writing behavior, so it is not a substitute for the no-write workflow.

## Display thresholds are not scientific cutoffs

Each species/field has independent lower and upper **log10 density** thresholds,
in the units shown for that field. For example,
`{'1': (2, None)}` keeps species 1 values with `log10(density) >= 2`; `None`
clears a bound. Zero, negative, and nonfinite densities are not log-plottable.

Thresholds hide points (and affected mesh elements); they do not clip density
values, change estimator input, or recompute densities. Reset/clear thresholds to
restore the valid points.

## Units and grain interpretation

Gas volume/column densities use `cm^-3` / `cm^-2`. Grain fields stay in SI:

| Grain quantity | 3D field | 2D field |
| --- | --- | --- |
| Number | `m^-3` | `m^-2` |
| Mass | `kg m^-3` | `kg m^-2` |
| Geometric cross-section (`area`) | `m^-1` | dimensionless |

Grain LOS is **projected 3D density, not an integrated column density**, and
requires `dimension=3`. Geometric cross-section density is **not optical depth**;
it does not include an extinction efficiency or radiative-transfer calculation.
Mixed views preserve each field's own units rather than applying gas conversions
to grains. See also the [grain guide](grains.md).
