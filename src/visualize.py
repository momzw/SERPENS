import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.parameters import Parameters
from scipy.ndimage.filters import gaussian_filter
from matplotlib.widgets import Slider, RangeSlider

matplotlib.use('TkAgg')
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 18})
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amssymb}')


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
            'colormap': matplotlib.colormaps["afmhot"],
            'lim': 20,
            'singlePlot': False,
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
        params = Parameters()
        self.sim = rebsim
        self.ps = rebsim.particles
        self.source_is_moon = params.int_spec['moon']
        self.fc = self._init_celestial_colors()

        self._init_figure()

    def _init_figure(self):
        params = Parameters()
        ns = params.num_species

        # colorbar: dpi: 800, facecolor w, colors at colorbar k
        self.fig = plt.figure(figsize=(self.vis_params['figsize'], self.vis_params['figsize']),
                              dpi=self.vis_params['dpi'])
        self.fig.patch.set_facecolor('k')

        if not self.vis_params['singlePlot'] and ns > 1:
            self.subplot_rows = int(np.ceil(ns / 3))
            self.subplot_columns = params.num_species if ns <= 3 else 3
            self.single = False
        else:
            self.subplot_rows = 1
            self.subplot_columns = 1
            self.single = True

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

        self.boundary = params.int_spec["r_max"] * self.sim.particles["source"].orbit(
            primary=self.sim.particles["source_primary"]).a

    def _init_celestial_colors(self):
        if len(self.vis_params['celest_colors']) == 0:
            fc = ['yellow', 'sandybrown', 'yellow']
        else:
            if isinstance(self.vis_params['celest_colors'], list):
                fc = self.vis_params['celest_colors']
                while len(fc) < self.sim.N_active:
                    fc.append('yellow')
            else:
                fc = ['yellow', 'sandybrown', 'yellow']
        return fc

    def setup_ax(self, ax):
        ax.set_aspect("equal")
        lim = self.vis_params['lim'] * self.ps["source_primary"].r

        if self.vis_params["perspective"] == "topdown":
            ax.set_xlabel("x-distance in planetary radii", fontsize=20, labelpad=15, color='w')
            ax.set_ylabel("y-distance in planetary radii", fontsize=20, labelpad=15, color='w')
        elif self.vis_params["perspective"] == "los":
            ax.set_xlabel("y-distance in planetary radii", fontsize=20, labelpad=8, color='w')
            ax.set_ylabel("z-distance in planetary radii", fontsize=20, labelpad=15, color='w')
        else:
            raise ValueError("Invalid perspective in plotting.")

        primary_coord1, primary_coord2 = self._get_coordinates("source_primary")

        ax.set_xlim([-lim + primary_coord1, lim + primary_coord1])
        ax.set_ylim([-lim + primary_coord2, lim + primary_coord2])

        loc_num = self.vis_params['lim'] + 1
        xlocs = np.linspace(-lim + primary_coord1, lim + primary_coord1, loc_num)
        ylocs = np.linspace(-lim + primary_coord2, lim + primary_coord2, loc_num)
        xlabels = np.around((np.array(xlocs) - primary_coord1) / self.ps["source_primary"].r, 1)
        ylabels = np.around((np.array(ylocs) - primary_coord2) / self.ps["source_primary"].r, 1)

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
        # ax.xaxis.label.set_color('white')
        # ax.yaxis.label.set_color('white')

        # ax.set_xlabel('')
        # ax.set_ylabel('')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        if self.vis_params["perspective"] == 'los':
            ax.invert_xaxis()

        if self.vis_params["perspective"] == "topdown":
            if self.vis_params['shadow_polygon']:
                self._add_shadow_polygon(ax)
            if self.vis_params['planetstar_connection']:
                self._add_planetstar_connection(ax)

            line_color = self.fc[Parameters.int_spec["source_index"] - 1]
            op = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=["source"], color=line_color,
                                   primary=self.ps["source_primary"])
            op.particles.set_color(line_color)
            op.particles.set_sizes([0])

        self._add_patches(ax)
        self._add_additional_celestials(ax)

    def _get_coordinates(self, obj_str):
        valid_obj = ['source', 'source_primary']
        if obj_str in valid_obj:
            if self.vis_params["perspective"] == "topdown":
                return self.ps[obj_str].x, self.ps[obj_str].y
            elif self.vis_params["perspective"] == "los":
                return -self.ps[obj_str].y, self.ps[obj_str].z
            else:
                pass

    def _add_patches(self, ax):

        if self.vis_params['show_primary']:
            fc = self.fc[1] if Parameters.int_spec["source_index"] > 2 else self.fc[0]
            coord1, coord2 = self._get_coordinates("source_primary")
            primary_patch = plt.Circle((coord1, coord2), self.ps["source_primary"].r, fc=fc, zorder=10, label="primary")
            ax.add_patch(primary_patch)

        if self.vis_params['show_source']:
            fc = self.fc[Parameters.int_spec["source_index"] - 1]
            coord1, coord2 = self._get_coordinates("source")
            source_patch = plt.Circle((coord1, coord2), self.ps["source"].r, fc=fc, ec='k',
                                      label="source", zorder=10, fill=True)
            ax.add_patch(source_patch)

        if self.source_is_moon and self.vis_params['show_hill']:
            coord1, coord2 = self._get_coordinates("source_primary")
            hill_patch = plt.Circle((coord1, coord2), self.ps["source_primary"].rhill, fc='green', fill=False)
            ax.add_patch(hill_patch)

    def _add_shadow_polygon(self, ax):
        assert self.vis_params["perspective"] == "topdown"
        apex = np.asarray(self.ps[1].xyz) * (1 + self.ps[1].r / (self.ps[0].r - self.ps[1].r))

        orthogonal_vector_to_pos = np.array([-self.ps[1].y, self.ps[1].x, 0]) / np.linalg.norm(
            np.array([-self.ps[1].y, self.ps[1].x, 0]))
        left_flank = self.ps[1].r * orthogonal_vector_to_pos + np.asarray(self.ps[1].xyz)
        right_flank = - self.ps[1].r * orthogonal_vector_to_pos + np.asarray(self.ps[1].xyz)

        t1 = plt.Polygon([apex[:2], left_flank[:2], right_flank[:2]], color='black', alpha=0.3, zorder=10)
        ax.add_patch(t1)

    def _add_planetstar_connection(self, ax):
        assert self.vis_params["perspective"] == "topdown"
        ax.plot([self.ps[0].x, self.ps[1].x], [self.ps[0].y, self.ps[1].y], color='bisque',
                linestyle=':', linewidth=1, zorder=10)

    def _add_additional_celestials(self, ax):
        # Additional celestial objects
        number_additional_celest = self.sim.N_active - 3 if self.source_is_moon else self.sim.N_active - 2
        if number_additional_celest > 0:
            moons_indices = [i for i in range(2, self.sim.N_active)]
            if not Parameters.int_spec["source_index"] == 2:
                moons_indices.remove(Parameters.int_spec["source_index"] - 1)

            if self.vis_params["perspective"] == 'topdown':
                op_add = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=moons_indices,
                                           color=self.fc[moons_indices[0]:], primary=self.ps[1],
                                           orbit_style="trail", lw=.5)
                op_add.particles.set_color(self.fc[moons_indices[0]:])
                op_add.particles.set_sizes([0 for _ in self.fc[moons_indices[0]:]])
            else:
                op_add = None

            for i in range(number_additional_celest):
                ind = moons_indices[i]
                if self.vis_params["perspective"] == "topdown":
                    op_add.orbits[i].set_alpha(.5)
                    major_obj = plt.Circle((self.ps[ind].x, self.ps[ind].y), self.ps[ind].r, alpha=0.7,
                                           fc=self.fc[ind], zorder=10)
                    ax.add_patch(major_obj)

                elif self.vis_params["perspective"] == "los":
                    major_obj = plt.Circle((-self.ps[ind].y, self.ps[ind].z), self.ps[ind].r, alpha=0.7,
                                           fc=self.fc[ind], zorder=10)
                    ax.add_patch(major_obj)


class Visualize(BaseVisualizer):

    # TODO: Check out Plotly as an alternative

    def __init__(self, rebsim, interactive=True, **kwargs):
        super().__init__(rebsim, **kwargs)

        self.cf = None
        self.c = None
        self.scatter = None
        self.cb_interact = None
        self.interactive = interactive

    def __call__(self, save_path=None, show_bool=True, **kwargs):

        handles, labels = self.axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        #legend = self.fig.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.75, 0.80), fontsize=25, labelcolor='white')
        #if len(by_label) == 0:
        #    legend.remove()

        #self.fig.text(0.2, 0.5, "y-distance in primary radii", rotation="vertical", verticalalignment='center',
        #              horizontalalignment='right', fontsize='x-large', transform=self.fig.transFigure)
        #self.fig.text(0.5, 0.05, "x-distance in primary radii", horizontalalignment='center', fontsize='x-large',
        #              transform=self.fig.transFigure)

        if save_path is not None:
            fn = kwargs.get("filename", -1)
            orbit_phase = np.around(self.sim.particles["source"].orbit(
                primary=self.sim.particles["source_primary"]).theta * 180 / np.pi)

            frame_identifier = f"SERPENS_{fn}"
            plt.savefig(f'output/{save_path}/plots/{frame_identifier}.png', bbox_inches='tight')
            print(f"\t plotted {fn}")
            if not show_bool:
                plt.close('all')
            #time.sleep(1)

        if show_bool:
            #if not len(by_label) == 0:
            #   legend.remove()
            #   self.fig.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.735, 0.83), fontsize=10)

            if self.cf is None and self.c is None and self.scatter is None:
                self.interactive = False

            if self.interactive and len(self.axs) == 1:

                slider_ax = self.fig.add_axes([0.9, 0.15, 0.03, 0.6])

                #divider = make_axes_locatable(self.axs[0])
                #slider_ax = divider.append_axes('right', size='3%', pad=3)

                if self.cf is not None:
                    slider = RangeSlider(slider_ax, "Threshold", self.cf.norm.vmin, self.cf.norm.vmax,
                                         orientation='vertical', facecolor='crimson')
                elif self.c is not None:
                    slider = RangeSlider(slider_ax, "Threshold", self.c.norm.vmin, self.c.norm.vmax,
                                         orientation='vertical', facecolor='crimson')
                else:
                    slider = RangeSlider(slider_ax, "Threshold", self.scatter.norm.vmin, self.scatter.norm.vmax,
                                         orientation='vertical', facecolor='crimson')

                if self.cf or self.c is not None:
                    axfreq = self.fig.add_axes([0.92, 0.15, 0.03, 0.6])
                    smoothing_slider = Slider(ax=axfreq, label='Smoothing', valmin=0.1, valmax=5, valinit=.8,
                                              orientation='vertical', facecolor='crimson')
                    smoothing_slider.on_changed(
                        lambda update: self.__update_interactive(update, slider, smoothing_slider))
                    slider.on_changed(lambda update: self.__update_interactive(update, slider, smoothing_slider))
                    smoothing_slider.label.set_rotation(90)
                    smoothing_slider.valtext.set_rotation(90)
                    smoothing_slider.label.set_fontsize(15)
                else:
                    slider.on_changed(lambda update: self.__update_interactive(update, slider))

                slider.valtext.set_rotation(90)
                slider.valtext.set_fontsize(12)
                slider.label.set_rotation(90)
                slider.label.set_fontsize(12)
                slider.label.set_color('white')

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

    def __update_interactive(self, val, slider=None, smoothing_slider=None):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        lvls = np.linspace(slider.val[0], slider.val[1], 25)

        if isinstance(self.vis_params["colormap"], list):
            cmap = self.vis_params["colormap"][0]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.vis_params["colormap"]
            cmap.set_bad(color='k', alpha=1.)

        if self.cf is not None:
            for tpcf in self.cf.collections:
                tpcf.remove()
            # Update the image's colormap
            np.seterr(divide='ignore')
            logdens = np.where(self.dens > 0,
                               np.log(gaussian_filter(self.dens, smoothing_slider.val, mode='constant')), 0)
            np.seterr(divide='warn')
            self.cf = self.axs[0].contourf(self.X, self.Y, logdens / np.log(10), levels=lvls, cmap=cmap,
                                           vmin=slider.val[0], vmax=slider.val[1], zorder=5, alpha=1)

        if self.c is not None:
            for tpc in self.c.collections:
                tpc.remove()
            np.seterr(divide='ignore')
            logdens = np.where(self.dens > 0,
                               np.log(gaussian_filter(self.dens, smoothing_slider.val, mode='constant')), 0)
            np.seterr(divide='warn')
            self.c = self.axs[0].contour(self.X, self.Y, logdens / np.log(10), levels=lvls, cmap=cmap,
                                         vmin=slider.val[0], vmax=slider.val[1], zorder=4, alpha=1)

        if self.cb_interact is not None:
            self.cb_interact.norm.vmin = slider.val[0]
            self.cb_interact.norm.vmax = slider.val[1]

        if self.scatter is not None:
            #self.scatter.set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

            logdens = self.scatlogd[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            x = self.scatx[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            y = self.scaty[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            xy = np.vstack((x, y))

            self.scatter.set_offsets(xy.T)
            self.scatter.set_array(logdens / np.log(10))
            self.scatter.set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def add_densityscatter(self, ax, x, y, density, d=3, **kwargs):
        self.vis_params.update(kwargs)

        if not self.single:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]
        self.setup_ax(ax_obj)

        logdens = np.where(density > 0, np.log(density), 0)

        self.scatx = x
        self.scaty = y
        self.scatlogd = logdens

        if isinstance(self.vis_params["colormap"], list):
            cmap = self.vis_params["colormap"][ax]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.vis_params["colormap"]
            cmap.set_bad(color='k', alpha=1.)

        self.scatter = ax_obj.scatter(x, y, c=logdens / np.log(10), cmap=cmap, vmin=self.vis_params["lvl_min"],
                                      vmax=self.vis_params["lvl_max"], s=.2, zorder=self.vis_params["zorder"])

        divider = make_axes_locatable(ax_obj)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cax.tick_params(axis='both', which='major', labelsize=20, color='w', colors='w')
        self.cb_interact = plt.colorbar(self.scatter, cax=cax, orientation='vertical',
                                        format=self.vis_params['cb_format'])
        self.cb_interact.ax.locator_params(nbins=12)
        if self.vis_params["perspective"] == 'los':
            self.cb_interact.ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')
        else:
            self.cb_interact.ax.set_title(fr'[cm$^{{{-d}}}$]', fontsize=22, loc='left', pad=20, color='w')

    def add_triplot(self, ax, x, y, simplices, trialpha=.8, **kwargs):
        self.vis_params.update(kwargs)

        ax = self.axs[ax]
        self.setup_ax(ax)
        ax.triplot(x, y, simplices, linewidth=0.1, c='w', zorder=self.vis_params["zorder"], alpha=trialpha)

    def add_colormesh(self, ax, X, Y, dens, contour=True, fill_contour=False, **kwargs):
        kw = {
            "cb_format": '%.2f',
            "logmin": 0,
            "logmax": 0,
            "zorder": 2,
            "cfilter_coeff": 1,
            "numlvls": 10,
            "celest_colors": 'default',
            "perspective": "topdown",
            "lvlmin": None,
            "lvlmax": None,
            "show_planet": True,
            "show_moon": True
        }
        kw.update(kwargs)

        if not self.single:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]
        self.setup_ax(ax_obj)

        self.X = X
        self.Y = Y
        self.dens = dens

        np.seterr(divide='ignore')
        logdens = np.where(dens > 0, np.log(gaussian_filter(dens, kw['cfilter_coeff'], mode='constant')), 0)

        lvlmin = np.min(np.log(dens)[np.log(dens) > 0]) if kw["lvlmin"] is None else kw["lvlmin"]
        lvlmax = np.max(np.log(dens)[np.log(dens) > 0]) if kw["lvlmax"] is None else kw["lvlmax"]
        lvlmin += kw['logmin']
        lvlmax -= kw['logmax']

        lvls = np.linspace(lvlmin, lvlmax, kw['numlvls']) / np.log(10)

        if isinstance(self.vis_params["colormap"], list):
            cmap = self.vis_params["colormap"][ax]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.vis_params["colormap"]
            cmap.set_bad(color='k', alpha=1.)

        np.seterr(divide='warn')

        if contour:
            self.c = ax_obj.contour(X, Y, logdens / np.log(10), cmap=cmap, levels=lvls,
                                    zorder=kw['zorder'])
        if fill_contour:
            self.cf = ax_obj.contourf(X, Y, logdens / np.log(10), cmap=cmap, levels=lvls, zorder=kw['zorder'] - 1)

        divider = make_axes_locatable(ax_obj)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.tick_params(axis='both', which='major', labelsize=15)
        self.cb_interact = plt.colorbar(self.cf, cax=cax, orientation='vertical', format=kw['cb_format'])
        self.cb_interact.ax.set_title(r' log N [cm$^{-2}$]', fontsize=18, loc='left', pad=12)

    def empty(self, ax):
        if not self.single:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]
        self.setup_ax(ax_obj)