"""Local, single-dataset Dash application. Dash is an optional dependency."""

import argparse
import socket
import uuid
from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.parameters import GLOBAL_PARAMETERS
from .data import DisplayOptions, PhaseData, PlotRequest, VisualizationDataProvider
from .figures import build_figure, empty_figure


class DashboardController:
    """Keep scientific arrays on the server, not in browser stores."""

    def __init__(self, analyzer, cache_size=8):
        self.provider = VisualizationDataProvider(analyzer, cache_size)
        self._results = OrderedDict()
        self.cache_size = cache_size

    def prepare(self, request):
        with self.provider._lock:
            data = self.provider.prepare(request)
            token = uuid.uuid4().hex
            self._results[token] = data
            while len(self._results) > self.cache_size:
                self._results.popitem(last=False)
            fields = []
            for field in getattr(data, 'fields', ()):
                valid = field.valid & field.visible
                values = np.log10(field.density[valid])
                fields.append(dict(key=field.key, name=field.name, units=field.units,
                                   minimum=float(values.min()) if len(values) else None,
                                   maximum=float(values.max()) if len(values) else None))
            return dict(token=token, fields=fields, phase=isinstance(data, PhaseData), view=request.view)

    def _result(self, token):
        if token not in self._results:
            raise ValueError('Prepared view expired. Change a selection or refresh the archive.')
        return self._results[token]

    def render(self, token, display):
        with self.provider._lock:
            return build_figure(self._result(token), display)

    def phase_csv(self, token):
        with self.provider._lock:
            data = self._result(token)
            if not isinstance(data, PhaseData):
                raise ValueError('Prepare a phase curve before downloading CSV.')
            return data.frame.to_csv(index=False)

    def refresh(self):
        with self.provider._lock:
            self._results.clear()
            self.provider.refresh()


def figure_html(figure):
    return pio.to_html(go.Figure(figure), include_plotlyjs=True, full_html=True,
                       config={'responsive': True, 'displaylogo': False})


def create_app(analyzer):
    """Construct a Dash app without starting servers or creating output files."""
    try:
        from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
        from dash.exceptions import PreventUpdate
    except ImportError as exc:
        raise ImportError("Install the optional dashboard dependency: pip install 'serpens[web]'") from exc

    app = Dash(__name__, title='SERPENS visualization', suppress_callback_exceptions=True)
    background = pio.templates['plotly_dark'].layout.paper_bgcolor
    foreground = pio.templates['plotly_dark'].layout.font.color
    app.index_string = app.index_string.replace('</head>', f'''<style>
        html, body {{ margin: 0; min-height: 100%; background: {background}; color: {foreground}; }}
        .Select {{ color: #111; }}
    </style></head>''')
    controller = DashboardController(analyzer)
    app._visualization_controller = controller
    species = [dict(label=s.name, value=s.id) for s in GLOBAL_PARAMETERS.get('all_species', [])]
    populations = [dict(label=f'Population {p}', value=int(p)) for p in getattr(analyzer, 'grain_populations', {})]
    initial = max(0, len(analyzer.sa) - 1)

    def control(label, child):
        return html.Div([html.Label(label), child], style={'marginBottom': '12px'})

    def number(identifier, value=None, **kwargs):
        return dcc.Input(id=identifier, type='number', value=value, debounce=True,
                         style={'width': '100%'}, **kwargs)

    app.layout = html.Div([
        html.H2('SERPENS · local visualization'),
        #html.P('One trusted dataset per process. Density thresholds hide points; clearing restores all valid data.'),
        html.Div([
            html.Div([
                control('View', dcc.Dropdown(id='view', options=[
                    {'label': label, 'value': value} for label, value in
                    [('Planar', 'planar'), ('Line of sight', 'los'), ('3D', '3d'),
                     ('1D profile cut', 'profile'), ('Phase curves', 'phase')]], value='planar', clearable=False)),
                control('Archive timestep', number('timestep', initial, min=0, max=initial, step=1)),
                html.Button('Refresh archive', id='refresh', n_clicks=0),
                control('Fields', dcc.Dropdown(id='kind', options=['gas', 'grain', 'mixed'],
                                               value='gas' if species else 'grain', clearable=False)),
                control('Gas species', dcc.Dropdown(id='species', options=species,
                                                   value=[s['value'] for s in species], multi=True)),
                html.Div(control('Estimator dimension', dcc.RadioItems(id='dimension', options=[
                    {'label': '2D column (xy)', 'value': 2}, {'label': '3D volume', 'value': 3}], value=3)), id='dimension-controls'),
                html.Div([
                    control('Grain quantity', dcc.Dropdown(id='quantity', options=['number', 'mass', 'area'], value='number', clearable=False)),
                    control('Populations (empty = all together)', dcc.Dropdown(id='populations', options=populations, value=[], multi=True)),
                    control('Minimum grain radius [m]', number('radius-min', min=0)),
                    control('Maximum grain radius [m]', number('radius-max', min=0)),
                    html.P('LOS shows projected 3D grain density, not integrated column density. Geometric area is not optical depth.')
                ], id='grain-controls'),
                html.Div(control('Source name or hash (blank = default/all)', dcc.Input(id='source', type='text', value='', debounce=True)), id='source-controls'),
                html.Div(control('Maximum cut distance [source radii]', number('distance', 6, min=1)), id='profile-controls'),
                html.Div([
                    control('Phase input', dcc.RadioItems(id='phase-input', options=['archive', 'csv'], value='archive')),
                    control('Existing local phase CSV', dcc.Input(id='phase-file', value='phase-curve.csv', type='text', debounce=True)),
                    control('Last source orbits', number('orbits', 1, min=0.01)),
                    html.Button('Compute / load phase curve', id='compute-phase', n_clicks=0),
                    control('Statistic', dcc.RadioItems(id='statistic', options=['max', 'mean'], value='max')),
                    dcc.Checklist(id='phase-curves', options=[{'label': 'Column', 'value': 'column'},
                                                            {'label': 'Volume', 'value': 'volume'}], value=['column', 'volume'])
                ], id='phase-controls'),
                html.H3('Display'),
                control('Colors', dcc.Dropdown(id='colors', options=['Viridis', 'Plasma', 'Inferno', 'Cividis', 'YlOrBr', 'Turbo'], value='Viridis', clearable=False)),
                control('Extent [primary radii] (blank = auto)', number('extent', min=0.01)),
                dcc.Checklist(id='display', options=[{'label': label, 'value': value} for label, value in [
                    ('Separate 2D panels (one at a time)', 'separate'), ('Scatter', 'scatter'), ('Bodies', 'bodies'),
                    ('Planar orbits', 'orbits'), ('Triangulation', 'mesh'), ('Planar shadow', 'shadow'),
                    ('Star connection', 'connection'), ('Star in 3D', 'star'), ('Log profile axis', 'log')]],
                    value=['separate', 'scatter', 'bodies', 'orbits']),
                control('Mesh opacity', dcc.Slider(id='mesh-opacity', min=0, max=1, step=0.05, value=0.3)),
                html.H3('Log10 density thresholds'),
                html.Button('Reset thresholds', id='reset', n_clicks=0),
                html.Div(id='threshold-controls'),
            ], style={'flex': '0 1 320px', 'minWidth': '270px', 'padding': '16px', 'boxSizing': 'border-box',
                      'background': background}),
            html.Div([
                dcc.Loading([html.Div(id='preparation-status', role='status'), dcc.Store(id='prepared')]),
                html.Div(id='render-status', role='status', style={'color': '#a02020'}),
                html.Div([
                    html.Button('← Previous', id='previous-field', n_clicks=0),
                    dcc.Dropdown(id='field-picker', clearable=False,
                                 style={'minWidth': '160px', 'flex': '1 1 180px'}),
                    html.Span(id='field-counter'),
                    html.Button('Next →', id='next-field', n_clicks=0),
                ], id='field-navigation', style={'display': 'none'}),
                dcc.Loading(html.Div(
                    dcc.Graph(id='graph', figure=empty_figure('Preparing data…'), responsive=True,
                              style={'width': '100%', 'height': '100%'},
                              config={'responsive': True, 'displaylogo': False,
                                      'toImageButtonOptions': {'format': 'png', 'filename': 'serpens'}}),
                    style={'width': '100%', 'height': 'min(82vh, 1000px)', 'minHeight': '520px'})),
                html.Button('Download HTML', id='download-html', n_clicks=0),
                html.Button('Download phase CSV', id='download-csv', n_clicks=0, disabled=True),
                dcc.Download(id='html-download'), dcc.Download(id='csv-download')
            ], style={'flex': '1 1 600px', 'minWidth': 0}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px', 'alignItems': 'flex-start'}),
    ], style={'fontFamily': 'system-ui, sans-serif', 'padding': '16px', 'boxSizing': 'border-box',
              'width': '100%',
              'background': background, 'color': foreground})

    @app.callback(Output('grain-controls', 'style'), Output('profile-controls', 'style'),
                  Output('phase-controls', 'style'), Output('source-controls', 'style'),
                  Output('dimension-controls', 'style'), Input('view', 'value'), Input('kind', 'value'))
    def conditional_controls(view, kind):
        visible = lambda condition: {} if condition else {'display': 'none'}
        return (visible(kind != 'gas' and view not in ('profile', 'phase')), visible(view == 'profile'),
                visible(view == 'phase'), visible(view in ('profile', 'phase') or kind != 'gas'), visible(view == 'planar'))

    @app.callback(Output('prepared', 'data'), Output('preparation-status', 'children'),
                  Output('timestep', 'max'),
                  Input('view', 'value'), Input('timestep', 'value'), Input('kind', 'value'),
                  Input('species', 'value'), Input('dimension', 'value'), Input('quantity', 'value'),
                  Input('populations', 'value'), Input('radius-min', 'value'), Input('radius-max', 'value'),
                  Input('source', 'value'), Input('distance', 'value'), Input('phase-input', 'value'),
                  Input('phase-file', 'value'), Input('orbits', 'value'), Input('compute-phase', 'n_clicks'),
                  Input('refresh', 'n_clicks'))
    def prepare(view, timestep, kind, selected, dimension, quantity, population_ids, rmin, rmax,
                source, distance, phase_input, phase_file, orbits, compute, refresh):
        try:
            if ctx.triggered_id == 'refresh':
                controller.refresh()
            maximum = max(0, len(analyzer.sa) - 1)
            if view == 'phase' and ctx.triggered_id != 'compute-phase':
                return None, 'Choose phase inputs, then click Compute / load phase curve.', maximum
            if timestep is None or int(timestep) != timestep:
                raise ValueError('Choose an integer archive timestep.')
            radius_range = None
            if kind != 'gas' and (rmin is not None or rmax is not None):
                if rmin is None or rmax is None:
                    raise ValueError('Set both grain radius bounds, or clear both.')
                radius_range = (rmin, rmax)
            request = PlotRequest(int(timestep), view, tuple(selected or []),
                                  dimension if view == 'planar' else 3,
                                  source_hash=source.strip() or None, max_distance_rp=distance,
                                  kind=kind, quantity=quantity,
                                  population_ids=tuple(population_ids) if population_ids else None,
                                  radius_range=radius_range,
                                  phase_file=phase_file if view == 'phase' and phase_input == 'csv' else None,
                                  orbits=orbits)
            metadata = controller.prepare(request)
            return metadata, '', maximum
        except Exception as exc:
            return None, f'Cannot prepare this view: {exc}', max(0, len(analyzer.sa) - 1)

    @app.callback(Output('threshold-controls', 'children'), Output('download-csv', 'disabled'),
                  Input('prepared', 'data'), Input('reset', 'n_clicks'))
    def threshold_controls(prepared, reset):
        rows = []
        for field in (prepared or {}).get('fields', []):
            label = f"{field['name']} [{field['units']}]"
            bounds = ('No valid densities' if field['minimum'] is None else
                      f"Valid log range: {field['minimum']:.4g} to {field['maximum']:.4g}")
            rows.append(html.Div([html.Strong(label), html.Div(bounds),
                                  dcc.Input(id={'type': 'lower', 'key': field['key']}, type='number', value=None,
                                            placeholder='Minimum log10', debounce=True, style={'width': '48%'}),
                                  dcc.Input(id={'type': 'upper', 'key': field['key']}, type='number', value=None,
                                            placeholder='Maximum log10', debounce=True, style={'width': '48%'})]))
        return rows, not bool((prepared or {}).get('phase'))

    @app.callback(Output('field-picker', 'options'), Output('field-picker', 'value'),
                  Output('field-navigation', 'style'), Output('field-counter', 'children'),
                  Input('prepared', 'data'), Input('display', 'value'),
                  Input('previous-field', 'n_clicks'), Input('next-field', 'n_clicks'),
                  State('field-picker', 'value'))
    def navigate_fields(prepared, flags, previous, next_clicks, current):
        fields = (prepared or {}).get('fields', [])
        options = [{'label': field['name'], 'value': field['key']} for field in fields]
        keys = [option['value'] for option in options]
        if not keys:
            return options, None, {'display': 'none'}, ''
        if ctx.triggered_id == 'prepared' or current not in keys:
            current = keys[0]
        elif ctx.triggered_id in ('previous-field', 'next-field'):
            offset = -1 if ctx.triggered_id == 'previous-field' else 1
            current = keys[(keys.index(current) + offset) % len(keys)]
        visible = (prepared or {}).get('view') in ('planar', 'los') and 'separate' in (flags or []) and len(keys) > 1
        style = {'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'flexWrap': 'wrap'} if visible else {'display': 'none'}
        return options, current, style, f'{keys.index(current) + 1} / {len(keys)}'

    @app.callback(Output('graph', 'figure'), Output('render-status', 'children'),
                  Input('prepared', 'data'), Input({'type': 'lower', 'key': ALL}, 'value'),
                  Input({'type': 'upper', 'key': ALL}, 'value'), Input('colors', 'value'),
                  Input('extent', 'value'), Input('display', 'value'), Input('mesh-opacity', 'value'),
                  Input('statistic', 'value'), Input('phase-curves', 'value'), Input('field-picker', 'value'),
                  State({'type': 'lower', 'key': ALL}, 'id'), State({'type': 'upper', 'key': ALL}, 'id'))
    def render(prepared, lower, upper, colors, extent, flags, opacity, statistic, curves, active_field, lower_ids, upper_ids):
        if not prepared:
            return empty_figure('Choose a valid selection. Phase curves require explicit preparation.'), ''
        try:
            lows = {identifier['key']: value for identifier, value in zip(lower_ids, lower)}
            highs = {identifier['key']: value for identifier, value in zip(upper_ids, upper)}
            thresholds = {key: (lows.get(key), highs.get(key)) for key in set(lows) | set(highs)}
            display = DisplayOptions(thresholds=thresholds, colorscale=colors, extent=extent,
                                     active_field=active_field,
                                     separate='separate' in flags, scatter='scatter' in flags, bodies='bodies' in flags,
                                     mesh='mesh' in flags, mesh_opacity=opacity, orbits='orbits' in flags,
                                     shadow='shadow' in flags, connection='connection' in flags,
                                     show_star='star' in flags, log_scale='log' in flags, statistic=statistic,
                                     column_density='column' in curves, particle_density='volume' in curves)
            return controller.render(prepared['token'], display), ''
        except Exception as exc:
            return empty_figure('Unable to render the selected options.'), str(exc)

    @app.callback(Output('html-download', 'data'), Input('download-html', 'n_clicks'),
                  State('graph', 'figure'), prevent_initial_call=True)
    def download_html(clicks, figure):
        if not clicks:
            raise PreventUpdate
        return dict(content=figure_html(figure), filename='serpens.html', type='text/html')

    @app.callback(Output('csv-download', 'data'), Input('download-csv', 'n_clicks'),
                  State('prepared', 'data'), prevent_initial_call=True)
    def download_csv(clicks, prepared):
        if not clicks or not prepared or not prepared.get('phase'):
            raise PreventUpdate
        try:
            return dict(content=controller.phase_csv(prepared['token']), filename='serpens-phase.csv', type='text/csv')
        except ValueError:
            return no_update

    app._visualization_callbacks = dict(prepare=prepare, render=render, thresholds=threshold_controls,
                                        navigate=navigate_fields, conditional=conditional_controls,
                                        html=download_html, csv=download_csv)
    return app


def _check_port(port):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError('Port must be an integer between 1 and 65535.')
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is unavailable on 127.0.0.1; choose another port or stop the existing server.') from exc


def launch_notebook(analyzer, port=8050, jupyter_mode='external'):
    """Launch once per analyzer. Repeated notebook cells return the same app."""
    if jupyter_mode not in ('external', 'inline'):
        raise ValueError('Choose external or inline notebook display.')
    existing = getattr(analyzer, '_dashboard_app', None)
    if existing is not None:
        if existing._dashboard_port != port:
            raise ValueError('This analyzer already has a dashboard on another port. Reuse it or restart the kernel.')
        print(f'Reusing SERPENS dashboard: http://127.0.0.1:{port}')
        return existing
    _check_port(port)
    app = create_app(analyzer)
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, jupyter_mode=jupyter_mode)
    app._dashboard_port = port
    analyzer._dashboard_app = app
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local SERPENS dashboard (run from the dataset directory).')
    parser.add_argument('--reference-system', default=None)
    parser.add_argument('--port', default=8050, type=int)
    args = parser.parse_args(argv)
    try:
        _check_port(args.port)
        from src.serpens_analyzer import SerpensAnalyzer
        analyzer = SerpensAnalyzer(reference_system=args.reference_system, save_output=False, save_archive=False)
        app = create_app(analyzer)
        app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'SERPENS dashboard: {exc}\n')


if __name__ == '__main__':
    main()