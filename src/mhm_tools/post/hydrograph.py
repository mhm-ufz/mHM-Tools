"""
26.01.2023
"""
import itertools
import logging
from pathlib import Path

import hydroeval as he
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class Hydrograph:
    levels = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }
    grid = None
    logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
    logger = logging.getLogger("hydrograph")
    plots = (0, 0, 0, 0)

    def __init__(self, log_level):
        self.logger.setLevel(self.levels[log_level])

    def get_row_col(self):
        # finds the first unused gridcell for the next plot
        for i, row in enumerate(self.grid):
            for j, v in enumerate(row):
                if not v:
                    return i, j
        msg = "No unused gridcell left"
        raise ValueError(msg)

    def is_last_plot(self, n):
        # finds the first unused gridcell for the next plot
        return np.sum(self.plots[n:]) == 1

    def get_catchment_area(self, path, ndecimal=0):
        config_file = Path(path + "ConfigFile.log")
        if config_file.exists():
            doc = config_file.open("r").readlines()
            for line in doc[::-1]:
                if line.strip():
                    return (
                        f"{float(line.replace('Total[km2]', '').strip()):.{ndecimal}f}"
                    )
        self.logger.warning("Area could not be read.")
        return ""

    def get_long_time_monthly_mean(self, variable):
        """Takes a variable and calculated the long time average value for every month of the year
        :param variable: xarray with a variable and a time
        :return: list of twelve numbers corresponding to the long term average value for each month of the year
        """
        var_ses = [[], [], [], [], [], [], [], [], [], [], [], []]
        for i in range(len(variable)):
            var_ses[int(variable.time[i].dt.month.data) - 1].append(variable[i])
        return np.array([np.mean(np.array(m)) for m in var_ses])

    def plot_on_axis(
        self, function, xvalues, yvalues: list, colors=None, labels=None, **arguments
    ):
        """
        Plots multiple graphs for specified function.
        :param function: matplotlib plot function e.g. ax.plot, plt.plot, ax.scatter, ax.errorbar, ...
        :param xvalues: list of x values
        :param yvalues: list of arrays with the y values
        :param colors: optional list of colors (default red, blue)
        :param labels: optional list of labels  (default simulated discharge, observed discharge)
        :param arguments: other plot relevant arguemts corresponding to the given input function. e.g. linewidth=0.5 for plt.plot
        """
        if labels is None:
            labels = ["simulated discharge", "observed discharge"]
        if colors is None:
            colors = ["#EF4340", "#3A4F99"]
        for i, yvalue in enumerate(yvalues):
            arguments["color"] = colors[i]
            arguments["label"] = labels[i]
            function(xvalues, yvalue, **arguments)

    def check_which_plots_to_create(self, a):
        """
        creates all possible permutations of 4 different plots and tests the plot_code (sum(a_i * 2*i) for a_i = 0 or 1)
        against them to create a touple indicating which plots to produce
        """
        possible_permutations = list(itertools.product([0, 1], repeat=4))
        for permutation in possible_permutations:
            check = 0
            for i, v in enumerate(permutation):
                check += v * 2**i
            if int(a) == check:
                self.logger.debug(f"plots to be produced: {permutation}")
                self.plots = permutation
                return
        self.logger.warning("No plots will be produced since none were specified.")

    def gen_hydrograph(self, input_path, filename, show, save, title, plot_code):
        """
        Read in discharge data and plot the simulated against the observed discharge
        for different time resolutions and a seasonality as well as plotting simulated against observed discharge.
        :param input_path: Path to discharge.nc file
        :param filename: Filename of the resulting file. e.g. hydrograph.png
        :param show: bool if plots should be shown or not
        :param save: bool if plots should be saved or not
        :param title: title given to the hydrograph
        :param plot_code: code indicating which plots to create
        """

        if input_path[-1] != "/":
            input_path += "/"
        if (
            len(filename.split("/")) == 1
        ):  # by default the hydrograph is saved to the data directory
            filename = input_path + filename
        self.check_which_plots_to_create(plot_code)
        if sum(self.plots) == 0:
            return
        with xr.open_dataset(input_path + "discharge.nc") as ds:
            discharge_timestep = ds.load()
            for v in discharge_timestep.variables:
                if not isinstance(v, str):
                    msg = f"variable name is not a string - {v} - {type(v)}"
                    raise TypeError(msg)
                for key in ["sim", "obs"]:
                    if key in v:
                        catchment = str(int(v.split("_")[1]))
                        discharge_timestep = discharge_timestep.rename({v: key})
            # discharge_monthly = discharge_timestep.resample(time="M").mean(skipna=False)
            discharge_yearly = discharge_timestep.resample(time="Y").mean(skipna=False)

            nse_timestep = he.evaluator(
                he.nse, discharge_timestep["sim"], discharge_timestep["obs"]
            )
            kge_timestep, r_timestep, alpha_timestep, beta_timestep = he.evaluator(
                he.kge, discharge_timestep["sim"], discharge_timestep["obs"]
            )
            # nse_monthly = he.evaluator(
            #     he.nse, discharge_monthly["sim"], discharge_monthly["obs"]
            # )
            # kge_monthly, r_monthly, alpha_monthly, beta_monthly = he.evaluator(
            #     he.kge, discharge_monthly["sim"], discharge_monthly["obs"]
            # )
            nse_yearly = he.evaluator(
                he.nse, discharge_yearly["sim"], discharge_yearly["obs"]
            )
            kge_yearly, r_yearly, alpha_yearly, beta_yearly = he.evaluator(
                he.kge, discharge_yearly["sim"], discharge_yearly["obs"]
            )

            fig = plt.figure(figsize=(7, 8))
            # fig.set_size_inches(7, 8)
            nrows = sum(self.plots) // 2 + 1
            ncols = 2
            self.grid = [
                False * ncols
            ] * nrows  # generate a grid indicating used and unused cells
            self.logger.debug(f"nrows = {nrows} and ncols = {ncols}")
            # gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
            gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1])
            area = self.get_catchment_area(input_path, ndecimal=0)
            fig.text(
                s=f"{catchment}",
                x=0.01,
                y=0.97,
                horizontalalignment="left",
                fontsize="x-large",
            )
            if area:
                fig.text(
                    s="Area = " + area + r"$km^2$",
                    x=0.5,
                    y=0.97,
                    horizontalalignment="center",
                    fontsize="x-large",
                )
            fig.suptitle(
                t=f"\n{title}\n",
                x=0.5,
                y=0.97,
                horizontalalignment="center",
                fontsize="x-large",
            )

            if self.plots[0]:
                self.logger.info("generating discharge plot")
                r, c = self.get_row_col()
                ax1 = fig.add_subplot(gs[r, c:])
                self.grid[r] = [True for c in self.grid[r]]
                self.logger.debug(self.grid)
                self.plot_on_axis(
                    function=ax1.scatter,
                    xvalues=discharge_timestep["time"],
                    yvalues=[discharge_timestep["sim"], discharge_timestep["obs"]],
                    s=1.0,
                )
                self.plot_on_axis(
                    function=ax1.plot,
                    xvalues=discharge_timestep["time"],
                    yvalues=[discharge_timestep["sim"], discharge_timestep["obs"]],
                    linewidth=0.3,
                )
                ax1.legend()
                ax1.set_title(
                    f"KGE = {kge_timestep[0]:.2f}, "
                    f"NSE = {nse_timestep[0]:.2f}, "
                    f"alpha = {alpha_timestep[0]:.2f}, "
                    f"beta = {beta_timestep[0]:.2f}, "
                    f"r = {r_timestep[0]:.2f}",
                    horizontalalignment="center",
                )  # , alpha_day={alpha_timestep[0]:.2f} , beta_day={beta_timestep[0]:.2f} , r_day={r_timestep[0]:.2f}')
                ax1.set_ylabel(r"Q $[m^3 s^{-1}]$")
                ax1.set_xlim(
                    discharge_timestep["time"][0], discharge_timestep["time"][-1]
                )

            # if self.plots[1]:
            #     self.logger.info('generating monthly discharge plot')
            #     r, c = self.get_row_col()
            #     if r == 0:
            #         ax2 = fig.add_subplot(gs[r,c:])
            #         self.grid[r] = [True for c in self.grid[r]]
            #     else:
            #         ax2 = fig.add_subplot(gs[r, c])
            #         self.grid[r][c] = True
            #     self.logger.debug(self.grid)
            #     # time_years = [str(y.dt.year.data) for y in discharge_monthly["time"]]
            #     time_datetime = discharge_monthly["time"]
            #     # print(type(discharge_monthly['time'].values[0]), discharge_monthly['time'].values[0])
            #     # time_datetime = discharge_monthly['time'].astype(datetime.datetime)
            #     # print(type(discharge_monthly['time'].values[0].astype(datetime.datetime)))
            #     self.plot_on_axis(
            #         function=ax2.scatter,
            #         xvalues=time_datetime,
            #         yvalues=[discharge_monthly["sim"], discharge_monthly["obs"]],
            #         s=1
            #     )
            #     self.plot_on_axis(
            #         function=ax2.plot,
            #         xvalues=time_datetime,
            #         yvalues=[discharge_monthly["sim"], discharge_monthly["obs"]],
            #         linewidth=.3
            #     )
            #     ax2.set_title(
            #         f"Monthly:  KGE = {kge_monthly[0]:.2f} , NSE = {nse_monthly[0]:.2f}",
            #         horizontalalignment="center"
            #     )
            #     if r == 0:
            #         ax2.legend()
            #     # xfmt = mdates.DateFormatter('%')
            #     # ax2.xaxis.set_major_formatter(xfmt)
            #     ax2.xaxis.set_major_formatter(
            #         mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
            #     # ax2.xaxis.set_tick_params() #.locator_params(axis='x', nbins=9//column)
            #     # ax2.set_xticks(discharge_monthly['time'] // 2)
            #     ax2.set_ylabel(r"Q $[m^3 s^{-1}]$")
            #     # ax2.set_xlim(time_years[0], time_years[-1])
            #     # print(time_years)
            #     # tick_freq = len(time_years)/12 // 3 * 12 if len(time_years) > 3*12 else 12
            #     # ax2.set_xticks(time_years[::int(tick_freq)])

            if self.plots[1]:
                self.logger.info("generating yearly discharge plot")
                r, c = self.get_row_col()
                if r == 0 or self.is_last_plot(1):
                    ax3 = fig.add_subplot(gs[r, c:])
                    self.grid[r] = [True for c in self.grid[r]]
                else:
                    ax3 = fig.add_subplot(gs[r, c])
                    self.grid[r][c] = True
                self.logger.debug(self.grid)
                time_yearly = [str(y.dt.year.data) for y in discharge_yearly["time"]]
                self.plot_on_axis(
                    function=ax3.scatter,
                    xvalues=time_yearly,
                    yvalues=[discharge_yearly["sim"], discharge_yearly["obs"]],
                    s=1.0,
                )
                self.plot_on_axis(
                    function=ax3.plot,
                    xvalues=time_yearly,
                    yvalues=[discharge_yearly["sim"], discharge_yearly["obs"]],
                    linewidth=0.3,
                )
                if r == 0:
                    ax3.legend()
                ax3.set_title(
                    f"Yearly:  KGE = {kge_yearly[0]:.2f} , NSE = {nse_yearly[0]:.2f}",
                    horizontalalignment="center",
                )
                ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")
                ax3.set_xlim(time_yearly[0], time_yearly[-1])
                ax3.set_xticks(time_yearly[:: len(time_yearly) // 3])

            if self.plots[2]:
                self.logger.info("generating discharge seasonality plot")
                r, c = self.get_row_col()
                if r == 0 or self.is_last_plot(2):
                    ax4 = fig.add_subplot(gs[r, c:])
                    self.grid[r] = [True for c in self.grid[r]]
                else:
                    ax4 = fig.add_subplot(gs[r, c])
                    self.grid[r][c] = True
                self.logger.debug(self.grid)
                self.plot_on_axis(
                    function=ax4.scatter,
                    xvalues=range(1, 13),
                    yvalues=[
                        self.get_long_time_monthly_mean(discharge_timestep["sim"]),
                        self.get_long_time_monthly_mean(discharge_timestep["obs"]),
                    ],
                    s=1,
                )
                self.plot_on_axis(
                    function=ax4.plot,
                    xvalues=np.arange(1, 13),
                    yvalues=[
                        self.get_long_time_monthly_mean(discharge_timestep["sim"]),
                        self.get_long_time_monthly_mean(discharge_timestep["obs"]),
                    ],
                    linewidth=0.3,
                )
                if r == 0:
                    ax4.legend()
                ax4.set_title("Seasonality", horizontalalignment="center")
                ax4.set_xlim(0.5, 12.5)
                ax4.set_ylabel(r"Q $[m^3 s^{-1}]$")
                ax4.set_xlabel(r"month")
            if self.plots[3]:
                self.logger.info("generating discharge scatter plot")
                r, c = self.get_row_col()
                if r == 0 or self.is_last_plot(3):
                    self.logger.debug("scatter is last")
                    ax5 = fig.add_subplot(gs[r, c:])
                    self.grid[r] = [True for col in self.grid[r][c:]]
                else:
                    ax5 = fig.add_subplot(gs[r, c])
                    self.grid[r][c] = True
                self.logger.debug(self.grid)
                # add linear regression line to scatterplot
                self.plot_on_axis(
                    function=ax5.scatter,
                    xvalues=discharge_timestep["obs"],
                    yvalues=[discharge_timestep["sim"]],
                    s=50.0,
                    colors=["black"],
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.1,
                )
                m, b = np.polyfit(
                    discharge_timestep["obs"], discharge_timestep["sim"], deg=1
                )
                self.plot_on_axis(
                    function=ax5.plot,
                    xvalues=np.array([0, 1e6]),
                    yvalues=[m * np.array([0, 1e6]) + b, np.array([0, 1e6])],
                    colors=["red", "black"],
                    linewidth=0.5,
                )
                if r == 0:
                    ax5.legend()
                ax5.set_title(
                    "relation simulated to observed discharge",
                    horizontalalignment="center",
                )
                ax5.set_xlabel("Qobs $[m^3 s^{-1}]$")  # X Achsenbeschriftung
                ax5.set_ylabel("Qsim $[m^3 s^{-1}]$")  # Y Achsenbeschriftung
                lim = (
                    np.max(
                        [
                            np.max(discharge_timestep["obs"]),
                            np.max(discharge_timestep["sim"]),
                        ]
                    )
                    * 1.1
                )
                lim = lim - lim % 5 if lim - lim % 5 >= lim / 1.1 else lim + 5 - lim % 5
                ax5.set_xlim(0, lim)
                ax5.set_ylim(0, lim)
            plt.tight_layout()
            if save:
                fig.savefig(filename, bbox_inches="tight")
                self.logger.info(f"saved hydrograph to '{filename}'")
            if show:
                plt.show()
            else:
                plt.close()
