import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters import Parameters
from scipy.ndimage.filters import gaussian_filter
from matplotlib.widgets import Slider, RangeSlider
matplotlib.use('TkAgg')


class Visualize:

    # TODO (FOR LATER): Check out VISPY for fast interactive plots running on GPU

    def __init__(self, rebsim, interactive=True, cmap=plt.cm.afmhot, lim=35, singlePlot=False):
        params = Parameters()
        self.ns = params.num_species
        self.sim = rebsim
        self.ps = rebsim.particles
        self.moon = params.int_spec["moon"]

        if not singlePlot and self.ns > 1:
            self.subplot_rows = int(np.ceil(self.ns / 3))
            self.subplot_columns = params.num_species if self.ns <= 3 else 3
            self.single = False
        else:
            self.subplot_rows = 1
            self.subplot_columns = 1
            self.single = True

        self.fig = plt.figure(figsize=(15, 15))
        gs1 = gridspec.GridSpec(self.subplot_rows, self.subplot_columns)
        gs1.update(wspace=0.2, hspace=0.1)

        self.axs = [plt.subplot(gs1[f]) for f in range(self.subplot_rows * self.subplot_columns)]
        for ax_num in range(len(self.axs)):
            if ax_num >= params.num_species:
                self.axs[ax_num].remove()
            else:
                species_name = params.get_species(num=ax_num + 1).description
                self.axs[ax_num].set_facecolor('k')
                self.axs[ax_num].set_title(f"{species_name}", c='k', size=12, pad=15)

        if self.moon:
            self.fig.suptitle("Serpens Simulation around Planetary Body")
            self.boundary = params.int_spec["r_max"] * rebsim.particles["moon"].calculate_orbit(
                primary=rebsim.particles["planet"]).a
        else:
            self.fig.suptitle("Serpens Simulation around Stellar Body")
            self.boundary = params.int_spec["r_max"] * rebsim.particles["planet"].a

        self.cmap = cmap
        self.lim = lim

        self.cf = None
        self.c = None
        self.scatter = None
        self.cb_interact = None
        self.interactive = interactive

    def __call__(self, save_path=None, show_bool=True, **kwargs):

        handles, labels = self.axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        legend = self.fig.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.79, 0.83), fontsize=16)
        if len(by_label) == 0:
            legend.remove()

        #self.fig.text(0.2, 0.5, "y-distance in primary radii", rotation="vertical", verticalalignment='center',
        #              horizontalalignment='right', fontsize='x-large', transform=self.fig.transFigure)
        #self.fig.text(0.5, 0.05, "x-distance in primary radii", horizontalalignment='center', fontsize='x-large',
        #              transform=self.fig.transFigure)

        if save_path is not None:
            fn = kwargs.get("filename", -1)
            if self.moon:
                orbit_phase = np.around(self.sim.particles["moon"].calculate_orbit(
                    primary=self.sim.particles["planet"]).theta * 180 / np.pi)
            else:
                orbit_phase = np.around(self.sim.particles["planet"].calculate_orbit(
                    primary=self.sim.particles[0]).theta * 180 / np.pi)
            frame_identifier = f"SERPENS_{fn}_{orbit_phase}"
            plt.savefig(f'output/{save_path}/plots/{frame_identifier}.png', bbox_inches='tight')
            print(f"\t plotted {fn}")
            if not show_bool:
                plt.close('all')
            time.sleep(1)

        if show_bool:
            if not len(by_label) == 0:
                legend.remove()
                #self.fig.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.735, 0.83), fontsize=10)

            if self.cf is None and self.c is None and self.scatter is None:
                self.interactive = False

            if self.interactive and len(self.axs) == 1:

                slider_ax = self.fig.add_axes([0.9, 0.15, 0.03, 0.6])

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

                slider.label.set_rotation(90)
                slider.valtext.set_rotation(90)
                slider.label.set_fontsize(10)

                plt.show()
            else:
                plt.show()

    def __del__(self):
        plt.clf()
        plt.close()

    def set_title(self, title_string, size='xx-large'):
        self.fig.suptitle(title_string, size=size)

    def __setup_ax(self, ax, perspective, **kwargs):
        kw = {
            "show_planet": True,
            "show_moon": True,
            "show_hill": False,
            "celest_colors": []
        }
        kw.update(kwargs)

        ax.set_aspect("equal")
        lim = self.lim * self.ps["planet"].r if self.moon else self.lim * self.ps[0].r

        if perspective == "topdown":
            ps_star_coord1 = self.ps["star"].x
            ps_star_coord2 = self.ps["star"].y
            ps_planet_coord1 = self.ps["planet"].x
            ps_planet_coord2 = self.ps["planet"].y

            ax.set_xlabel("x-distance in planetary radii", fontsize=15, labelpad=15)
            ax.set_ylabel("y-distance in planetary radii", fontsize=15, labelpad=15)

        elif perspective == "los":
            ps_star_coord1 = -self.ps["star"].y
            ps_star_coord2 = self.ps["star"].z
            ps_planet_coord1 = -self.ps["planet"].y
            ps_planet_coord2 = self.ps["planet"].z

            ax.set_xlabel("y-distance in planetary radii", fontsize=15, labelpad=8)
            ax.set_ylabel("z-distance in planetary radii", fontsize=15, labelpad=15)

        else:
            raise ValueError("Invalid perspective in plotting.")

        ax.set_xlim([-lim + ps_planet_coord1, lim + ps_planet_coord1]) if self.moon else ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim + ps_planet_coord2, lim + ps_planet_coord2]) if self.moon else ax.set_ylim([-lim, lim])

        if len(kw['celest_colors']) == 0:
            fc = ['yellow', 'sandybrown', 'yellow']
        else:
            if isinstance(kw['celest_colors'], list):
                fc = kw['celest_colors']
                while len(fc) < self.sim.N_active:
                    fc.append('yellow')
            else:
                fc = ['yellow', 'sandybrown', 'yellow']

        if self.moon:
            ps_moon_coord1 = self.ps["moon"].x
            ps_moon_coord2 = self.ps["moon"].y

            if perspective == "los":
                ps_moon_coord1 = -self.ps["moon"].y
                ps_moon_coord2 = self.ps["moon"].z

            xlocs = np.linspace(ps_planet_coord1 - lim, ps_planet_coord1 + lim, 13)
            ylocs = np.linspace(ps_planet_coord2 - lim, ps_planet_coord2 + lim, 13)
            xlabels = np.around((np.array(xlocs) - ps_planet_coord1) / self.ps["planet"].r, 1)
            ylabels = np.around((np.array(ylocs) - ps_planet_coord2) / self.ps["planet"].r, 1)

            if kw['show_moon']:
                moon_patch = plt.Circle((ps_moon_coord1, ps_moon_coord2), self.ps["moon"].r, fc=fc[2], alpha=.7, label="exomoon", zorder=10)
                ax.add_patch(moon_patch)

            # Show direction to star and shadow:
            if perspective == "topdown":
                ax.plot([ps_star_coord1, ps_moon_coord1], [ps_star_coord2, ps_moon_coord2], color='bisque',
                        linestyle=':', linewidth=1, zorder=10)

                apex = np.asarray(self.ps["planet"].xyz) * (1 + self.ps["planet"].r / (self.ps["star"].r - self.ps["planet"].r))
                orthogonal_vector_to_pos = np.array([-ps_planet_coord2, ps_planet_coord1, 0]) / np.linalg.norm(np.array([-ps_planet_coord2, ps_planet_coord1, 0]))
                left_flank = self.ps["planet"].r * orthogonal_vector_to_pos + np.asarray(self.ps["planet"].xyz)
                right_flank = - self.ps["planet"].r * orthogonal_vector_to_pos + np.asarray(self.ps["planet"].xyz)

                t1 = plt.Polygon([apex[:2], left_flank[:2], right_flank[:2]], color='black', alpha=0.7, zorder=10)
                ax.add_patch(t1)

                #star_patch = plt.Circle((ps_star_coord1, ps_star_coord2), self.ps[0].r, fc=fc[0], zorder=10, label="star")
                #ax.add_patch(star_patch)

        else:
            xlocs = np.linspace(-lim, lim, 13)
            ylocs = np.linspace(-lim, lim, 13)
            xlabels = np.around(np.array(xlocs) / self.ps[0].r, 2)
            ylabels = np.around(np.array(ylocs) / self.ps[0].r, 2)

            star_patch = plt.Circle((ps_star_coord1, ps_star_coord2), self.ps[0].r, fc=fc[0], zorder=10, label="star")
            ax.add_patch(star_patch)

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        if kw['show_planet']:
            planet_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), self.ps["planet"].r, fc=fc[1], ec='k',
                                      label="exoplanet", zorder=10, fill=True)
            ax.add_patch(planet_patch)
        if kw['show_hill']:
            hill_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), self.ps["planet"].rhill, fc='green',
                                    fill=False)
            ax.add_patch(hill_patch)

        if perspective == "topdown":
            line_color = fc[2] if self.moon else fc[1]
            op = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=[2], color=line_color,
                                   primary=self.ps["planet"])
            op.particles.set_color(line_color)
            op.particles.set_sizes([0])

            #rect = plt.Rectangle((-lim + ps_planet_coord1, ps_planet_coord2 - self.ps["planet"].r), lim, 2*self.ps["planet"].r, alpha=.8, zorder=2, fc='black')
            #ax.add_patch(rect)

        # Additional celestial objects
        numberAdditionalCelest = self.sim.N_active - 3 if self.moon else self.sim.N_active - 2
        additionalCelestFirstIndex = 3 if self.moon else 2
        if numberAdditionalCelest > 0:

            celestial_indices = [i+additionalCelestFirstIndex for i in range(numberAdditionalCelest)]
            op_add = rebound.OrbitPlot(self.sim, fig=self.fig, ax=ax, particles=celestial_indices,
                                       color=fc[additionalCelestFirstIndex:], primary=self.ps["planet"])
            op_add.particles.set_color(fc[additionalCelestFirstIndex:])
            op_add.particles.set_sizes([0 for _ in fc[additionalCelestFirstIndex:]])

            for i in range(numberAdditionalCelest):

                if perspective == "topdown":
                    major_obj = plt.Circle((self.ps[additionalCelestFirstIndex + i].x,
                                            self.ps[additionalCelestFirstIndex + i].y),
                                           self.ps[additionalCelestFirstIndex + i].r, alpha=0.7,
                                           fc=fc[additionalCelestFirstIndex + i], zorder=10)
                    ax.add_patch(major_obj)

                elif perspective == "los":
                    major_obj = plt.Circle((-self.ps[additionalCelestFirstIndex + i].y,
                                            self.ps[additionalCelestFirstIndex + i].z),
                                           self.ps[additionalCelestFirstIndex + i].r, alpha=0.7,
                                           fc=fc[additionalCelestFirstIndex + i], zorder=10)
                    ax.add_patch(major_obj)

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', visible=True)
        if ax == self.axs[0]:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=True)
        else:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=False)
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        #ax.set_xlabel('')
        #ax.set_ylabel('')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        if perspective == 'los':
            ax.invert_xaxis()

    def __update_interactive(self, val, slider=None, smoothing_slider=None):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        lvls = np.linspace(slider.val[0], slider.val[1], 25)

        if isinstance(self.cmap, list):
            cmap = self.cmap[0]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.cmap
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
                                           vmin=slider.val[0], vmax=slider.val[1], zorder=1)

        if self.c is not None:
            for tpc in self.c.collections:
                tpc.remove()
            np.seterr(divide='ignore')
            logdens = np.where(self.dens > 0,
                               np.log(gaussian_filter(self.dens, smoothing_slider.val, mode='constant')), 0)
            np.seterr(divide='warn')
            self.c = self.axs[0].contour(self.X, self.Y, logdens / np.log(10), levels=lvls, cmap=cmap,
                                         vmin=slider.val[0], vmax=slider.val[1], zorder=1)

        if self.cb_interact is not None:
            self.cb_interact.norm.vmin = slider.val[0]
            self.cb_interact.norm.vmax = slider.val[1]

        if self.scatter is not None:
            #self.scatter.set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

            logdens = self.scatlogd[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            x = self.scatx[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            y = self.scaty[(slider.val[0] < self.scatlogd/np.log(10)) & (self.scatlogd/np.log(10) < slider.val[1])]
            xy = np.vstack((x,y))

            self.scatter.set_offsets(xy.T)
            self.scatter.set_array(logdens / np.log(10))
            self.scatter.set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def add_densityscatter(self, ax, x, y, density, perspective, **kwargs):
        kw = {
            "cb_format": '%.2f',
            "zorder": 1,
            "cfilter_coeff": 1,
            "vmin": None,
            "vmax": None,
            "celest_colors": 'default',
            "show_planet": True,
            "show_moon": True
        }
        kw.update(kwargs)

        if not self.single:
            ax_obj = self.axs[ax]
        else:
            ax_obj = self.axs[0]

        self.__setup_ax(ax_obj, perspective=perspective, celest_colors=kw["celest_colors"],
                        show_planet=kw["show_planet"], show_moon=kw["show_moon"])

        logdens = np.where(density > 0, np.log(density), 0)

        self.scatx = x
        self.scaty = y
        self.scatlogd = logdens

        if isinstance(self.cmap, list):
            cmap = self.cmap[ax]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.cmap
            cmap.set_bad(color='k', alpha=1.)

        self.scatter = ax_obj.scatter(x, y, c=logdens / np.log(10), cmap=cmap, marker='.', vmin=kw['vmin'],
                                      vmax=kw['vmax'], s=2, zorder=kw['zorder'])

        divider = make_axes_locatable(ax_obj)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.tick_params(axis='both', which='major', labelsize=12)
        self.cb_interact = plt.colorbar(self.scatter, cax=cax, orientation='vertical', format=kw['cb_format'])
        if perspective == 'los':
            self.cb_interact.ax.set_title(r'[cm$^{-2}$]', fontsize=12, loc='left', pad=12)
        else:
            self.cb_interact.ax.set_title(r'[cm$^{-3}$]', fontsize=12, loc='left', pad=12)

    def add_triplot(self, ax, x, y, simplices, perspective, **kwargs):
        kw = {
            "zorder": 1,
            "celest_colors": ['royalblue', 'sandybrown', 'yellow'],
            "alpha": .8
        }
        kw.update(kwargs)

        ax = self.axs[ax]
        self.__setup_ax(ax, perspective=perspective, celest_colors=kw["celest_colors"])
        ax.triplot(x, y, simplices, linewidth=0.1, c='w', zorder=kw["zorder"], alpha=kw["alpha"])

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

        self.__setup_ax(ax_obj, perspective=kw["perspective"], celest_colors=kw["celest_colors"],
                        show_planet=kw["show_planet"], show_moon=kw["show_moon"])

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

        if isinstance(self.cmap, list):
            cmap = self.cmap[ax]
            cmap.set_bad(color='k', alpha=1.)
        else:
            cmap = self.cmap
            cmap.set_bad(color='k', alpha=1.)

        np.seterr(divide='warn')

        if contour:
            self.c = ax_obj.contour(X, Y, logdens / np.log(10), cmap=cmap, levels=lvls,
                                    zorder=kw['zorder'])
        if fill_contour:
            self.cf = ax_obj.contourf(X, Y, logdens / np.log(10), cmap=cmap, levels=lvls, zorder=kw['zorder'] - 1)

        divider = make_axes_locatable(ax_obj)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.tick_params(axis='both', which='major', labelsize=12)
        self.cb_interact = plt.colorbar(self.cf, cax=cax, orientation='vertical', format=kw['cb_format'])
        self.cb_interact.ax.set_title(r'[cm$^{-2}$]', fontsize=12, loc='left', pad=12)