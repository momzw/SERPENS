import matplotlib as mpl

# Constant configurations for Matplotlib
try:
    FONT_CONFIG = {'family': 'serif', 'serif': ['Computer Modern'], 'size': 18}
    TEXT_CONFIG = {'usetex': True}
    TEX_CONFIG = {'preamble': r'\usepackage{amssymb}'}
    DEFAULT_FACECOLOR = 'yellow'

    # Setting the backend and configurations for Matplotlib
    mpl.use('TkAgg')
    mpl.rc('font', **FONT_CONFIG)
    mpl.rc('text', **TEXT_CONFIG)
    mpl.rc('text.latex', **TEX_CONFIG)
except Exception as exc:
    print(f"An exception occurred while trying to change matplotlib parameters: {exc}")
    pass

import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.parameters import Parameters
from matplotlib.widgets import Slider, RangeSlider


class ArgumentProcessor:

    def __init__(self, **kwargs):
        self.vis_params = kwargs
        self.process()

    def apply_defaults(self, default_values):
        for key, value in default_values.items():
            self.vis_params.setdefault(key, value)

    def process(self):
        # Set default values for missing keyword arguments
        default_values = {
            'colormap': mpl.colormaps["afmhot"],
            'lim': 20,
            "single_plot": False,
            "show_source": True,
            "show_primary": True,
            "show_hill": False,
            "celest_colors": ['yellow', 'sandybrown', 'yellow'],
            "cb_format": '%.2f',
            "cfilter_coeff": 1,
            "lvl_min": None,
            "lvl_max": None,
            "zorder": 1,
            "perspective": 'topdown',
            "figsize": 15,
            "dpi": 100,
            "shadow_polygon": True,
            "planetstar_connection": True,
            "mesh_smoothing": 1,
            "mesh_fill_contour": True,
            "mesh_contour": True
        }

        self.apply_defaults(default_values)


class BaseVisualizer(ArgumentProcessor):

    def __init__(self, rebsim, **kwargs):
        super().__init__(**kwargs)
        Parameters()
        self.sim = rebsim
        self.particles = rebsim.particles
        self.face_colors = self._init_celestial_colors()

        self._init_figure()

    def _get_primary(self, source_index) -> rebound.Particle:
        return self.sim.particles[
            rebound.hash(self.sim.particles[f"source{source_index}"].params['source_primary'])]

    def _init_figure(self):
        params = Parameters()
        ns = params.num_species

        # colorbar: dpi: 800, facecolor w, colors at colorbar k
        self.fig = plt.figure(figsize=(self.vis_params['figsize']*Parameters.num_species, self.vis_params['figsize']),
                              dpi=self.vis_params['dpi'])
        self.fig.patch.set_facecolor('k')

        if not self.vis_params['single_plot'] and ns > 1:
            self.subplot_rows = int(np.ceil(ns / 3))
            self.subplot_columns = params.num_species if ns <= 3 else 3
            self.single_plot = False
        else:
            self.subplot_rows = 1
            self.subplot_columns = 1
            self.single_plot = True

        gs1 = gridspec.GridSpec(self.subplot_rows, self.subplot_columns)
        gs1.update(wspace=0.2, hspace=0.1)

        self.axs = [plt.subplot(gs1[f]) for f in range(self.subplot_rows * self.subplot_columns)]
        for ax_num in range(len(self.axs)):
            if ax_num >= params.num_species:
                self.axs[ax_num].remove()
                self.axs[ax_num].grid(False)
            else:
                species_name = params.get_species(num=ax_num + 1).description
                self.axs[ax_num].set_facecolor('k')
                self.axs[ax_num].set_title(f"{species_name}", c='w', size=12, pad=15)

    def _init_celestial_colors(self):
        assert isinstance(self.vis_params['celest_colors'], list), "Please pass 'celest_colors' as a Python list."
        fc = self.vis_params['celest_colors']
        while len(fc) < self.sim.N_active:
            fc.append(DEFAULT_FACECOLOR)
        return fc

    def setup_ax(self, ax: plt.Axes) -> None:
        ax.set_aspect("equal")
        lim = self.vis_params['lim'] * self._get_primary(0).r

        if self.vis_params["perspective"] == "topdown":
            ax.set_xlabel("x-distance in planetary radii", fontsize=20, labelpad=15, color='w')
            ax.set_ylabel("y-distance in planetary radii", fontsize=20, labelpad=15, color='w')
        elif self.vis_params["perspective"] == "los":
            ax.set_xlabel("y-distance in planetary radii", fontsize=20, labelpad=8, color='w')
            ax.set_ylabel("z-distance in planetary radii", fontsize=20, labelpad=15, color='w')
        else:
            raise ValueError("Invalid perspective in plotting.")

        primary_coord1, primary_coord2 = self._get_coordinates_primary(source_index=0)

        ax.set_xlim([-lim + primary_coord1, lim + primary_coord1])
        ax.set_ylim([-lim + primary_coord2, lim + primary_coord2])

        loc_num = self.vis_params['lim'] + 1
        xlocs = np.linspace(-lim + primary_coord1, lim + primary_coord1, loc_num)
        ylocs = np.linspace(-lim + primary_coord2, lim + primary_coord2, loc_num)
        xlabels = np.around((np.array(xlocs) - primary_coord1) / self._get_primary(0).r, 1)
        ylabels = np.around((np.array(ylocs) - primary_coord2) / self._get_primary(0).r, 1)

        ax.set_xticks(xlocs[1:-1])
        ax.set_xticklabels([str(x) for x in xlabels][1:-1])
        ax.set_yticks(ylocs[1:-1])
        ax.set_yticklabels([str(y) for y in ylabels][1:-1])

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', visible=True)
        if ax == self.axs[0]:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=True)
        else:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=False)
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False)
        ax.tick_params(axis='both', which='major', labelsize=15, pad=10, colors='w')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        if self.vis_params["perspective"] == 'los':
            ax.invert_xaxis()

        if self.vis_params["perspective"] == "topdown":
            if self.vis_params['shadow_polygon']:
                self._add_shadow_polygon(ax)
            if self.vis_params['planetstar_connection']:
                self._add_planetstar_connection(ax)

            line_color = self.face_colors[2]
            op = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=["source0"], color=line_color,
                                   primary=self._get_primary(0))
            op.particles.set_color(line_color)
            op.particles.set_sizes([0])

        self._add_patches(ax)
        self._add_additional_celestials(ax)

    def _get_coordinates_source(self, source_index):
        obj_str = f"source{source_index}"
        if self.vis_params["perspective"] == "topdown":
            return self.particles[obj_str].x, self.particles[obj_str].y
        elif self.vis_params["perspective"] == "los":
            return -self.particles[obj_str].y, self.particles[obj_str].z
        else:
            pass

    def _get_coordinates_primary(self, source_index):
        primary = self._get_primary(source_index)
        if self.vis_params["perspective"] == "topdown":
            return primary.x, primary.y
        elif self.vis_params["perspective"] == "los":
            return -primary.y, primary.z
        else:
            pass

    def _add_patches(self, ax):

        if self.vis_params['show_primary']:
            fc_index = self.particles[rebound.hash(self.particles["source0"].params["source_primary"])].index
            fc = self.face_colors[fc_index]
            coord1, coord2 = self._get_coordinates_primary(source_index=0)
            primary_patch = plt.Circle((coord1, coord2), self._get_primary(0).r, fc=fc, zorder=10, label="primary")
            ax.add_patch(primary_patch)

        if self.vis_params['show_source']:
            fc = self.face_colors[self.particles["source0"].index]
            coord1, coord2 = self._get_coordinates_source(source_index=0)
            source_patch = plt.Circle((coord1, coord2), self.particles["source0"].r, fc=fc, ec='k',
                                      label="source", zorder=10, fill=True, alpha=0.7)
            ax.add_patch(source_patch)

        # if self.source_is_moon and self.vis_params['show_hill']:
        #    coord1, coord2 = self._get_coordinates("source_primary0")
        #    hill_patch = plt.Circle((coord1, coord2), self.ps["source_primary0"].rhill, fc='green', fill=False)
        #    ax.add_patch(hill_patch)

    def _add_shadow_polygon(self, ax):
        assert self.vis_params["perspective"] == "topdown"
        apex = np.asarray(self.particles[1].xyz) * (
                1 + self.particles[1].r / (self.particles[0].r - self.particles[1].r))

        orthogonal_vector_to_pos = np.array([-self.particles[1].y, self.particles[1].x, 0]) / np.linalg.norm(
            np.array([-self.particles[1].y, self.particles[1].x, 0]))
        left_flank = self.particles[1].r * orthogonal_vector_to_pos + np.asarray(self.particles[1].xyz)
        right_flank = - self.particles[1].r * orthogonal_vector_to_pos + np.asarray(self.particles[1].xyz)

        t1 = plt.Polygon([apex[:2], left_flank[:2], right_flank[:2]], color='black', alpha=0.3, zorder=10)
        ax.add_patch(t1)

    def _add_planetstar_connection(self, ax):
        assert self.vis_params["perspective"] == "topdown"
        ax.plot([self.particles[0].x, self.particles[1].x], [self.particles[0].y, self.particles[1].y], color='bisque',
                linestyle=':', linewidth=1, zorder=10)

    def _add_additional_celestials(self, ax):
        # Additional celestial objects
        source_is_moon = self.particles[rebound.hash(self.particles["source0"].params["source_primary"])].index > 0
        number_additional_celest = self.sim.N_active - 3 if source_is_moon else self.sim.N_active - 2
        if number_additional_celest > 0:
            moons_indices = [i for i in range(self.sim.N_active - number_additional_celest, self.sim.N_active)]
            if self.vis_params["perspective"] == 'topdown':
                op_add = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=moons_indices,
                                           color=self.face_colors[moons_indices[0]:], primary=self.particles[1],
                                           orbit_style="trail", lw=.5)
                op_add.particles.set_color(self.face_colors[moons_indices[0]:])
                op_add.particles.set_sizes([0 for _ in self.face_colors[moons_indices[0]:]])
            else:
                op_add = None

            for i in range(number_additional_celest):
                ind = moons_indices[i]
                if self.vis_params["perspective"] == "topdown":
                    op_add.orbits[i].set_alpha(.5)
                    major_obj = plt.Circle((self.particles[ind].x, self.particles[ind].y), self.particles[ind].r,
                                           alpha=0.7,
                                           fc=self.face_colors[ind], zorder=10)
                    ax.add_patch(major_obj)

                elif self.vis_params["perspective"] == "los":
                    major_obj = plt.Circle((-self.particles[ind].y, self.particles[ind].z), self.particles[ind].r,
                                           alpha=0.7,
                                           fc=self.face_colors[ind], zorder=10)
                    ax.add_patch(major_obj)


class Visualize(BaseVisualizer):

    def __init__(self, rebsim, interactive=True, **kwargs):
        super().__init__(rebsim, **kwargs)

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
            plt.savefig(f'output/{save_path}/plots/{frame_identifier}.png', bbox_inches='tight')
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
        logdens = np.where(density > 0, np.log10(density), 0)
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
            slider_ax = divider.append_axes('right', size='4%')
            self.slider_axs.append(slider_ax)

            cax = divider.append_axes('right', size='4%', pad=0.05)
            cax.tick_params(axis='both', which='major', labelsize=20, color='w', colors='w')
            self.colorbar_interact.append(plt.colorbar(scatter, cax=cax, orientation='vertical',
                                                       format=self.vis_params['cb_format']))
        else:
            if ax_index == 0:
                for i in range(Parameters.num_species):
                    slider_ax = divider.append_axes('right', size='4%')
                    self.slider_axs.append(slider_ax)

                    cax = divider.append_axes('right', size='4%', pad=0.05 * i)
                    cax.tick_params(axis='both', which='major', labelsize=20, color='w', colors='w')
                    self.colorbar_axs.append(cax)

            self.colorbar_interact.append(plt.colorbar(scatter, cmap=cmap, cax=self.colorbar_axs[ax_index], orientation='vertical',
                                                       format=self.vis_params['cb_format']))

        # Set colorbar parameters
        self.colorbar_interact[-1].ax.locator_params(nbins=12)
        if self.vis_params["perspective"] == 'los':
            self.colorbar_interact[-1].ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')
        else:
            self.colorbar_interact[-1].ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')

    def add_triplot(self, ax, x, y, simplices, trialpha=.8, **kwargs):
        self.vis_params.update(kwargs)

        ax = self.axs[ax]
        self.setup_ax(ax)
        ax.triplot(x, y, simplices, linewidth=0.1, c='w', zorder=self.vis_params["zorder"], alpha=trialpha)

    def empty(self, ax):
        if not self.single_plot:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]
        self.setup_ax(ax_obj)
