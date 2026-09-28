import os
import h5py
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rebound
from matplotlib.widgets import RangeSlider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots

import shutil

from src.parameters import GLOBAL_PARAMETERS
from .base import BaseVisualizer, grain_density_label, grain_log_density


class Visualize(BaseVisualizer):

    def __init__(self, rebsim, reference_system, interactive=True, **kwargs):
        super().__init__(rebsim, reference_system, **kwargs)

        self.cf = None
        self.c = None
        self.scatter = None
        self.colorbar_interact = []
        self.slider_axs = []
        self.colorbar_axs =  []
        self.scatters =  []
        self.scatter_axs = []
        self.interactive = interactive      # TODO: Need to fix non-interactive

    def __call__(self, save_path=None, show_bool=True, **kwargs):

        if save_path is not None:
            fn = kwargs.get("filename", -1)
            frame_identifier = f"SERPENS_{fn}"

            base_path = f'output/{save_path}/plots'
            if not os.path.exists(base_path):
                os.makedirs(f'output/{save_path}/plots', exist_ok=True)

            plt.savefig(os.path.join(base_path, f'{frame_identifier}.png'), bbox_inches='tight')
            print(f"\t plotted {fn}")
            if not show_bool:
                plt.close('all')

        if show_bool:
            if len(self.scatter_axs) == 0:
                self.interactive = False

            if self.interactive:
                sliders = []
                for slider_ax in self.slider_axs:
                    index = self.slider_axs.index(slider_ax)
                    _slider = RangeSlider(slider_ax, "Threshold", self.scatter_axs[index].norm.vmin, self.scatter_axs[index].norm.vmax,
                                         orientation='vertical', facecolor='crimson')
                    sliders.append(_slider)
                    sliders[index].on_changed(lambda update, s=sliders[index], ind=index: self.__update_interactive(update, s, ax_index=ind))

                    sliders[index].valtext.set_rotation(90)
                    sliders[index].valtext.set_fontsize(12)
                    sliders[index].label.set_rotation(90)
                    sliders[index].label.set_fontsize(12)
                    sliders[index].label.set_color('white')

                plt.show()
            else:
                plt.show()

    def __del__(self):
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    def set_title(self, title_string, size='xx-large', color='k'):
        self.fig.suptitle(title_string, size=size, c=color)

    def __update_interactive(self, _, slider=None, ax_index=0):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        if len(self.colorbar_interact) > 0:
            self.colorbar_interact[ax_index].norm.vmin = slider.val[0]
            self.colorbar_interact[ax_index].norm.vmax = slider.val[1]

        if len(self.scatter_axs) > 0:
            scatx = self.scatters[ax_index][0]
            scaty = self.scatters[ax_index][1]
            logdens = self.scatters[ax_index][2]

            logdens_window = logdens[(slider.val[0] < logdens) & (logdens < slider.val[1])]
            x = scatx[(slider.val[0] < logdens) & (logdens < slider.val[1])]
            y = scaty[(slider.val[0] < logdens) & (logdens < slider.val[1])]
            xy = np.vstack((x, y))

            self.scatter_axs[ax_index].set_offsets(xy.T)
            self.scatter_axs[ax_index].set_array(logdens_window)
            self.scatter_axs[ax_index].set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def add_densityscatter(self, ax_index: int, x, y, density, d=3, **kwargs):
        grain_quantity = kwargs.pop('grain_quantity', None)
        self.vis_params.update(kwargs)

        # Set up axes
        if not self.single_plot:
            ax_obj: plt.Axes = self.axs[ax_index]
            self.setup_ax(ax_obj)
        else:
            ax_obj: plt.Axes = self.axs[0]
            if ax_index == 0:
                self.setup_ax(ax_obj)
        divider = make_axes_locatable(ax_obj)

        # Get densities and append data to class list
        logdens = grain_log_density(density) if grain_quantity is not None else np.log10(density + 1e-5)
        self.scatters.append((x, y, logdens))

        # Set up colormaps
        if isinstance(self.vis_params["colormap"], list):
            cmap = self.vis_params["colormap"][ax_index]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.vis_params["colormap"]
            cmap.set_bad(color='k', alpha=1.)

        # Create axis scatter plot and append to class list
        scatter = ax_obj.scatter(x, y, c=logdens, cmap=cmap, vmin=self.vis_params["lvl_min"],
                                 vmax=self.vis_params["lvl_max"], s=.2, zorder=self.vis_params["zorder"])
        self.scatter_axs.append(scatter)

        # Create colorbar and slider axes based on single/non-single plot
        if not self.single_plot:
            if self.interactive:
                slider_ax = divider.append_axes('right', size='4%')
                self.slider_axs.append(slider_ax)

            cax = divider.append_axes('right', size='4%', pad=0.05)
            cax.tick_params(axis='both', which='major', labelsize=20, color='w', colors='w')
            self.colorbar_interact.append(plt.colorbar(scatter, cax=cax, orientation='vertical',
                                                       format=self.vis_params['cb_format']))
        else:
            if len(self.colorbar_axs) == 0:
                for i in range(self.num_species):
                    if self.interactive:
                        slider_ax = divider.append_axes('right', size='4%')
                        self.slider_axs.append(slider_ax)

                    cax = divider.append_axes('right', size='4%', pad=0.05 * i)
                    cax.tick_params(axis='both', which='major', labelsize=20, color='w', colors='w')
                    self.colorbar_axs.append(cax)

            self.colorbar_interact.append(
                plt.colorbar(scatter, cmap=cmap, cax=self.colorbar_axs[ax_index], orientation='vertical',
                             format=self.vis_params['cb_format'])
            )

        # Set colorbar parameters
        self.colorbar_interact[-1].ax.locator_params(nbins=12)
        if grain_quantity is not None:
            self.colorbar_interact[-1].set_label(grain_density_label(grain_quantity, d), color='w', usetex=False)
        elif self.vis_params["perspective"] == 'los':
            self.colorbar_interact[-1].ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')
        else:
            self.colorbar_interact[-1].ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')

    def add_triplot(self, ax_index, x, y, simplices, **kwargs):
        self.vis_params.update(kwargs)

        # Set up axes
        if not self.single_plot:
            ax_obj: plt.Axes = self.axs[ax_index]
            self.setup_ax(ax_obj)
        else:
            ax_obj: plt.Axes = self.axs[0]
            if ax_index == 0:
                self.setup_ax(ax_obj)

        ax_obj.triplot(x, y, simplices, linewidth=0.1, c='w', zorder=self.vis_params["zorder"], alpha=self.vis_params["trialpha"])

    def empty(self, ax):
        if not self.single_plot:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]
        self.setup_ax(ax_obj)


# TODO: WIP
class PlotlyVisualize(BaseVisualizer):

    def __init__(self, rebsim, reference_system, **kwargs):
        super().__init__(rebsim, reference_system, **kwargs)
        self._axis_prepared = set()

    def _get_coordinates_primary(self):
        from src.visualizing.figures import project_positions

        return tuple(project_positions([self._get_primary().xyz], self.vis_params['perspective'])[0])

    def _get_coordinates_source(self):
        from src.visualizing.figures import project_positions

        return tuple(project_positions([self.particles[self.reference_system].xyz],
                                       self.vis_params['perspective'])[0])

    def _init_figure(self):
        if not self.vis_params['single_plot'] and self.num_species > 1:
            self.subplot_rows = int(np.ceil(self.num_species / 3))
            self.subplot_columns = self.num_species if self.num_species <= 3 else 3
            self.single_plot = False
        else:
            self.subplot_rows = 1
            self.subplot_columns = 1
            self.single_plot = True

        self.fig = make_subplots(rows=self.subplot_rows,
                                 cols=self.subplot_columns,
                                 horizontal_spacing=0.08,
                                 vertical_spacing=0.08)
        self.fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='black',
            plot_bgcolor='black',
            showlegend=False,
            width=int(self.vis_params['figsize'] * self.subplot_columns * 260),
            height=int(self.vis_params['figsize'] * self.subplot_rows * 260),
        )

    def _subplot_position(self, ax_index):
        if self.single_plot:
            return 1, 1
        return ax_index // self.subplot_columns + 1, ax_index % self.subplot_columns + 1

    def _axis_suffix(self, row, col):
        axis_num = (row - 1) * self.subplot_columns + col
        if axis_num == 1:
            return '', ''
        return str(axis_num), str(axis_num)

    def _mpl_cmap_to_plotly(self, cmap):
        if isinstance(cmap, list):
            cmap = cmap[0]
        steps = np.linspace(0, 1, 12)
        colorscale = []
        for step in steps:
            rgba = cmap(step)
            rgb = (int(255 * rgba[0]), int(255 * rgba[1]), int(255 * rgba[2]))
            colorscale.append([float(step), f'rgb{rgb}'])
        return colorscale

    def _get_axis_bounds_and_labels(self):
        lim = self.vis_params['lim'] * self._get_primary().r
        primary_coord1, primary_coord2 = self._get_coordinates_primary()

        xlocs = np.linspace(-lim + primary_coord1, lim + primary_coord1, self.vis_params['lim'] + 1)
        ylocs = np.linspace(-lim + primary_coord2, lim + primary_coord2, self.vis_params['lim'] + 1)
        xlabels = np.around((np.array(xlocs) - primary_coord1) / self._get_primary().r, 1)
        ylabels = np.around((np.array(ylocs) - primary_coord2) / self._get_primary().r, 1)
        return lim, primary_coord1, primary_coord2, xlocs, ylocs, xlabels, ylabels

    def setup_ax(self, ax_index: int) -> None:
        row, col = self._subplot_position(ax_index)
        if (row, col) in self._axis_prepared:
            return

        lim, primary_coord1, primary_coord2, xlocs, ylocs, xlabels, ylabels = self._get_axis_bounds_and_labels()

        xaxis_title = "x-distance in primary radii"
        yaxis_title = "y-distance in primary radii"
        x_range = [-lim + primary_coord1, lim + primary_coord1]
        if self.vis_params["perspective"] == 'los':
            xaxis_title = "y-distance in primary radii"
            yaxis_title = "z-distance in primary radii"
            x_range.reverse()

        self.fig.update_xaxes(
            row=row,
            col=col,
            title_text=xaxis_title,
            range=x_range,
            autorange=False,
            tickmode='array',
            tickvals=xlocs[1:-1],
            ticktext=[str(x) for x in xlabels][1:-1],
            showgrid=False,
            zeroline=False,
            color='white',
        )
        self.fig.update_yaxes(
            row=row,
            col=col,
            title_text=yaxis_title,
            range=[-lim + primary_coord2, lim + primary_coord2],
            tickmode='array',
            tickvals=ylocs[1:-1],
            ticktext=[str(y) for y in ylabels][1:-1],
            showgrid=False,
            zeroline=False,
            scaleanchor=f'x{(row - 1) * self.subplot_columns + col if (row, col) != (1, 1) else ""}',
            scaleratio=1,
            color='white',
            autorange=False,
        )

        self._add_shapes(row, col)
        self._axis_prepared.add((row, col))

    def _add_circle(self, x_center, y_center, radius, color, row, col, alpha=0.7, line_color=None):
        from src.visualizing.figures import circle_trace

        self.fig.add_trace(
            circle_trace(x_center, y_center, radius, color, alpha=alpha, line_color=line_color),
            row=row,
            col=col,
        )

    def _add_shapes(self, row, col):
        if self.vis_params['show_primary']:
            primary = self._get_primary()
            fc = self.face_colors[primary.index]
            coord1, coord2 = self._get_coordinates_primary()
            self._add_circle(coord1, coord2, primary.r, fc, row, col, alpha=1.0)

        if self.vis_params['show_source']:
            source = self.particles[self.reference_system]
            fc = self.face_colors[source.index]
            coord1, coord2 = self._get_coordinates_source()
            self._add_circle(coord1, coord2, source.r, fc, row, col, alpha=0.7, line_color='black')

        if self.vis_params["perspective"] == "planar" and self.vis_params['planetstar_connection']:
            self.fig.add_trace(
                go.Scatter(
                    x=[self.particles[0].x, self.particles[1].x],
                    y=[self.particles[0].y, self.particles[1].y],
                    mode='lines',
                    line=dict(color='bisque', width=1, dash='dot'),
                    hoverinfo='skip',
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        self._add_additional_celestials_plotly(row, col)

    def _add_additional_celestials_plotly(self, row, col):
        from src.visualizing.figures import project_positions

        primary = self._get_primary()
        source_is_moon = primary.index > 0
        number_additional_celest = self.sim.N_active - 3 if source_is_moon else self.sim.N_active - 2
        if number_additional_celest <= 0:
            return

        moons_indices = [i for i in range(self.sim.N_active - number_additional_celest, self.sim.N_active)]
        for ind in moons_indices:
            x, y = project_positions([self.particles[ind].xyz], self.vis_params['perspective'])[0]
            self._add_circle(x, y, self.particles[ind].r, self.face_colors[ind], row, col, alpha=0.7)

    def __call__(self, save_path=None, show_bool=True, **kwargs):
        if save_path is not None:
            fn = kwargs.get("filename", -1)
            frame_identifier = f"SERPENS_{fn}"

            base_path = f'output/{save_path}/plots'
            if not os.path.exists(base_path):
                os.makedirs(base_path, exist_ok=True)

            try:
                self.fig.write_image(os.path.join(base_path, f'{frame_identifier}.png'))
            except Exception as exc:
                print(f"Could not export static Plotly image: {exc}")
            print(f"\t plotted {fn}")

        if show_bool:
            self.fig.show()

    def set_title(self, title_string, size='xx-large', color='black'):
        size_map = {'xx-large': 28, 'x-large': 24, 'large': 20}
        font_size = size_map.get(size, 22)
        self.fig.update_layout(title=dict(text=title_string, font=dict(size=font_size, color=color)))

    def add_densityscatter(self, ax_index: int, x, y, density, d=3, **kwargs):
        grain_quantity = kwargs.pop('grain_quantity', None)
        self.vis_params.update(kwargs)
        target_index = 0 if self.single_plot else ax_index
        row, col = self._subplot_position(target_index)
        self.setup_ax(target_index)

        logdens = grain_log_density(density) if grain_quantity is not None else np.log10(density + 1e-5)
        colorscale = self._mpl_cmap_to_plotly(
            self.vis_params["colormap"]) if not isinstance(self.vis_params["colormap"], list) else self._mpl_cmap_to_plotly(self.vis_params["colormap"][ax_index])

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=2,
                    color=logdens,
                    colorscale=colorscale,
                    cmin=self.vis_params["lvl_min"],
                    cmax=self.vis_params["lvl_max"],
                    colorbar=dict(title=grain_density_label(grain_quantity, d)
                                  if grain_quantity is not None else f'[cm^{{-{d}}}]'),
                    showscale=True,
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def add_triplot(self, ax_index, x, y, simplices, **kwargs):
        from src.visualizing.figures import mesh_trace

        self.vis_params.update(kwargs)
        target_index = 0 if self.single_plot else ax_index
        row, col = self._subplot_position(target_index)
        self.setup_ax(target_index)

        self.fig.add_trace(
            mesh_trace(x, y, simplices, opacity=self.vis_params["trialpha"]),
            row=row,
            col=col,
        )

    def empty(self, ax):
        target_index = 0 if self.single_plot else ax
        self.setup_ax(target_index)
