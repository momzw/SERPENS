"""Pure Plotly builders: no estimator calls, display, filesystem or global state."""

import colorsys
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import DisplayOptions, validate_phase_frame


def project_positions(positions, view):
    positions = np.asarray(positions)
    if view == 'los':
        return np.column_stack((-positions[:, 1], positions[:, 2]))
    return positions[:, :2].copy()


def circle_trace(x_center, y_center, radius, color, alpha=0.7, line_color=None):
    angles = np.linspace(0, 2 * np.pi, 100)
    return go.Scatter(x=x_center + radius * np.cos(angles), y=y_center + radius * np.sin(angles),
                      mode='lines', fill='toself', fillcolor=color, opacity=alpha,
                      line=dict(color=line_color or color, width=1), hoverinfo='skip', showlegend=False)


def mesh_trace(x, y, simplices, opacity=0.3, visible=None):
    simplices = np.asarray(simplices, dtype=int)
    edges = np.empty((0, 2), dtype=int)
    if simplices.size:
        edges = np.concatenate([simplices[:, pair] for pair in combinations(range(simplices.shape[1]), 2)])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        if visible is not None:
            edges = edges[np.all(np.asarray(visible)[edges], axis=1)]
    xs = np.full((len(edges), 3), np.nan)
    ys = xs.copy()
    xs[:, :2] = np.asarray(x)[edges]
    ys[:, :2] = np.asarray(y)[edges]
    return go.Scattergl(x=xs.ravel(), y=ys.ravel(), mode='lines', name='Triangulation',
                        line=dict(color='gray', width=0.5), opacity=opacity,
                        hoverinfo='skip', showlegend=False)


def density_selection(field, display):
    valid = field.valid & field.visible & np.isfinite(field.density) & (field.density > 0)
    logs = np.full(len(field.density), np.nan)
    logs[valid] = np.log10(field.density[valid])
    limits = (float(np.min(logs[valid])), float(np.max(logs[valid]))) if valid.any() else (0., 1.)
    if limits[0] == limits[1]:
        limits = (limits[0] - 0.5, limits[1] + 0.5)
    mask = valid.copy()
    bounds = display.thresholds.get(field.key)
    if bounds is not None:
        lower, upper = bounds
        if lower is not None and not np.isfinite(lower) or upper is not None and not np.isfinite(upper):
            raise ValueError('Thresholds must be finite or cleared')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError('Minimum threshold must not exceed maximum')
        if lower is not None:
            mask &= logs >= lower
        if upper is not None:
            mask &= logs <= upper
    return mask, logs, limits


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
    fig.update_layout(template='plotly_dark', autosize=True)
    return fig


def _field_colorscale(display, index):
    if index == 0:
        return display.colorscale
    alternatives = [scale for scale in ('Viridis', 'Plasma', 'Inferno', 'Cividis', 'YlOrBr', 'Turbo')
                    if scale != display.colorscale]
    if index <= len(alternatives):
        return alternatives[index - 1]
    hue = ((index - len(alternatives)) * 0.61803398875) % 1
    def shade(lightness):
        return 'rgb({},{},{})'.format(*(round(channel * 255) for channel in
                                      colorsys.hls_to_rgb(hue, lightness, 0.8)))
    return [[0, shade(0.16)], [1, shade(0.82)]]


def _field_colorbar(field, index, total):
    return dict(title=dict(text=f'{field.name}<br>log10 [{field.units}]', side='right'),
                len=0.85 / total, y=1 - (index + 0.5) / total, x=1.02)


def _overlays(fig, data, display, row, col):
    view = data.request.view
    for body in data.bodies:
        xy = project_positions(np.array([body.position]), view)[0]
        if display.bodies:
            trace = circle_trace(*xy, body.radius, 'gold' if body.star else 'royalblue')
            trace.name = 'Primary' if body.primary else body.name
            fig.add_trace(trace, row=row, col=col)
        if view == 'planar' and display.orbits and body.orbit is not None:
            xy = project_positions(body.orbit, view)
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], name='Orbit', mode='lines',
                                     line=dict(color='gray', width=1), showlegend=False,
                                     hoverinfo='skip'), row=row, col=col)
    if view != 'planar' or len(data.bodies) < 2:
        return
    star, planet = data.bodies[:2]
    if display.connection:
        xy = np.array([star.position, planet.position])
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], name='Star connection', mode='lines',
                                 line=dict(color='bisque', dash='dot'), hoverinfo='skip',
                                 showlegend=False), row=row, col=col)
    if display.shadow and star.radius > planet.radius:
        delta = planet.position[:2] - star.position[:2]
        distance = np.linalg.norm(delta)
        if distance > 0:
            normal = np.array([-delta[1], delta[0]]) / distance
            apex = planet.position[:2] + delta * planet.radius / (star.radius - planet.radius)
            xy = np.array([apex, planet.position[:2] + normal * planet.radius,
                           planet.position[:2] - normal * planet.radius, apex])
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], name='Shadow', mode='lines',
                                     fill='toself', fillcolor='gray', opacity=0.3,
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'), row=row, col=col)


def build_figure(data, display=None):
    display = display or DisplayOptions()
    if data.request.view == '3d':
        return build_3d_figure(data, display)
    if data.request.view == 'profile':
        return build_profile_figure(data, display)
    if data.request.view == 'phase':
        return build_phase_figure(data.frame, display.statistic, display.column_density, display.particle_density)
    if data.request.view not in ('planar', 'los'):
        raise ValueError('This builder requires planar or los data')
    if not data.fields:
        return empty_figure(data.message or 'No fields selected.')
    if display.extent is not None and (not np.isfinite(display.extent) or display.extent <= 0):
        raise ValueError('View extent must be positive, in primary radii')
    fields = data.fields
    if display.separate:
        fields = (next((field for field in fields if field.key == display.active_field), fields[0]),)
    title = f'{fields[0].name} [{fields[0].units}]' if display.separate else 'Combined fields'
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])
    for index, field in enumerate(fields):
        mask, logs, limits = density_selection(field, display)
        xy = project_positions(field.positions, data.request.view)
        if display.mesh:
            fig.add_trace(mesh_trace(*xy.T, field.simplices, opacity=display.mesh_opacity, visible=mask), row=1, col=1)
        if display.scatter:
            fig.add_trace(go.Scattergl(x=xy[mask, 0], y=xy[mask, 1], mode='markers', name=field.name,
                                      customdata=field.density[mask],
                                      hovertemplate=f'{field.name}<br>x=%{{x}} m<br>y=%{{y}} m<br>density=%{{customdata:.5g}} {field.units}<extra></extra>',
                                      marker=dict(size=3, color=logs[mask],
                                                  colorscale=_field_colorscale(display, index),
                                                  cmin=limits[0], cmax=limits[1],
                                                  colorbar=_field_colorbar(field, index, len(fields)),
                                                  showscale=True)), row=1, col=1)
        if not mask.any():
            fig.add_annotation(text=f'{field.name}: no data in selected range', showarrow=False,
                               x=0.5, y=0.5, xref='x domain', yref='y domain')
    _overlays(fig, data, display, 1, 1)
    los = data.request.view == 'los'
    center = project_positions(data.center[None, :], data.request.view)[0]
    xopts, yopts = {}, {}
    if display.extent is not None:
        lim = data.radius * display.extent
        xr = [center[0] - lim, center[0] + lim]
        xopts = dict(range=xr[::-1] if los else xr, autorange=False)
        yopts = dict(range=[center[1] - lim, center[1] + lim], autorange=False)
    else:
        xopts = dict(autorange='reversed' if los else True)
        yopts = dict(autorange=True)
    fig.update_xaxes(title_text='−y [m]' if los else 'x [m]', **xopts)
    fig.update_yaxes(title_text='z [m]' if los else 'y [m]', scaleanchor='x', scaleratio=1, **yopts)
    fig.update_layout(template='plotly_dark', autosize=True,
                      title=f'{data.request.view} · t={data.time:g} s',
                      margin=dict(l=65, r=170, t=80, b=60), uirevision=str(data.request))
    return fig


def build_3d_figure(data, display=None):
    display = display or DisplayOptions()
    fig = go.Figure()
    any_visible = False
    for index, field in enumerate(data.fields):
        mask, logs, limits = density_selection(field, display)
        any_visible |= mask.any()
        xyz = field.positions[mask]
        if display.scatter:
            fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                                      name=field.name, customdata=field.density[mask],
                                      hovertemplate=f'{field.name}<br>x=%{{x}} m<br>y=%{{y}} m<br>z=%{{z}} m<br>density=%{{customdata:.5g}} {field.units}<extra></extra>',
                                      marker=dict(size=2, opacity=0.4, color=logs[mask],
                                                  colorscale=_field_colorscale(display, index),
                                                  cmin=limits[0], cmax=limits[1],
                                                  colorbar=_field_colorbar(field, index, len(data.fields)),
                                                  showscale=True)))
    phi, theta = np.mgrid[0:2 * np.pi:50j, 0:np.pi:30j]
    for body in data.bodies:
        if not display.bodies or not (body.primary or body.star and display.show_star):
            continue
        x = body.position[0] + body.radius * np.sin(theta) * np.cos(phi)
        y = body.position[1] + body.radius * np.sin(theta) * np.sin(phi)
        z = body.position[2] + body.radius * np.cos(theta)
        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=np.zeros_like(x), showscale=False,
                                 colorscale='Hot' if body.star else 'Blues', name=body.name,
                                 hoverinfo='name'))
    if not any_visible:
        fig.add_annotation(text='No finite positive density in selected range.', showarrow=False,
                           x=0.5, y=0.5, xref='paper', yref='paper')
    scene = dict(aspectmode='data', xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='z [m]')
    if display.extent is not None:
        if not np.isfinite(display.extent) or display.extent <= 0:
            raise ValueError('View extent must be positive')
        for i, axis in enumerate(('xaxis', 'yaxis', 'zaxis')):
            scene[axis] = dict(range=[data.center[i] - display.extent * data.radius,
                                     data.center[i] + display.extent * data.radius])
    fig.update_layout(template='plotly_dark', autosize=True, scene=scene, margin=dict(r=170),
                      title=f'3D · t={data.time:g} s', uirevision=str(data.request))
    return fig


def build_profile_figure(data, display=None):
    display = display or DisplayOptions()
    if not data.profiles:
        return empty_figure(data.message or 'No profiles selected.')
    fig = go.Figure()
    for profile in data.profiles:
        for key, label in (('raw', 'raw'), ('smoothed', 'Gaussian σ=3')):
            values = np.asarray(profile[key]).copy()
            values[~np.isfinite(values) | ((values <= 0) & display.log_scale)] = np.nan
            fig.add_trace(go.Scatter(x=profile['radii'], y=values, mode='lines',
                                     name=f"{profile['name']} · {label}", connectgaps=False))
    fig.update_layout(template='plotly_dark', autosize=True, title='Source-relative column density cut',
                      xaxis_title='Distance [source radii]', yaxis_title='Density [cm^-2]',
                      yaxis_type='log' if display.log_scale else 'linear')
    return fig


def build_phase_figure(frame, statistic='max', column_density=True, particle_density=True):
    from scipy.interpolate import make_interp_spline

    frame = validate_phase_frame(frame)
    if statistic not in ('max', 'mean'):
        raise ValueError('Phase statistic must be max or mean')
    if frame.empty or not (column_density or particle_density):
        return empty_figure('No phase samples or density curves selected.')
    # Duplicate phases are combined for rendering only; exported samples stay intact.
    frame = frame.groupby('Phase', as_index=False).mean().sort_values('Phase')
    dimensions = ([3] if particle_density else []) + ([2] if column_density else [])
    fig = make_subplots(rows=len(dimensions), cols=1, shared_xaxes=True)
    phases = frame['Phase'].to_numpy()
    for row, dimension in enumerate(dimensions, 1):
        values = frame[f'{statistic.title()}_{dimension}D'].to_numpy()
        if len(phases) > 1:
            x = np.linspace(phases.min(), phases.max(), 200)
            y = make_interp_spline(phases, values, k=min(3, len(phases) - 1))(x)
        else:
            x, y = phases, values
        label = 'Column' if dimension == 2 else 'Volume'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines' if len(phases) > 1 else 'markers',
                                 name=f'{label} · {statistic}'), row=row, col=1)
        fig.update_yaxes(title_text=f'log10 density [cm^-{dimension}]', row=row, col=1)
    fig.update_xaxes(title_text='Phase [degrees]', range=[0, 360])
    fig.update_layout(template='plotly_dark', autosize=True, height=350 * len(dimensions),
                      title='Phase statistics (legacy logarithmic definition)')
    return fig