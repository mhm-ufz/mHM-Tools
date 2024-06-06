"""26.01.2023
"""

import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class Catchment:
    name = None
    area = None


class Objectives:
    kge = None
    nse = None
    alpha = None
    beta = None
    r = None
    diff = None
    rel_diff = None


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
    logger = logging.getLogger(__name__)
    plots = (0, 0, 0, 0)
    catchment = Catchment()
    discharge_data = None
    objectives = Objectives()

    def __init__(self, log_level):
        self.logger.setLevel(self.levels[log_level])

    @staticmethod
    def remove_empty_values(arr1, arr2):
        if len(arr1) != len(arr2):
            exeption = "The two timeseries do not have the same length."
            raise Exception(exeption)
        arr1_ret, arr2_ret = [], []
        for i, v in enumerate(arr1):
            w = arr2[i]
            if v is not None and v is not np.nan and w is not None and w is not np.nan:
                arr1_ret.append(v)
                arr2_ret.append(w)
        return np.array(arr1_ret), np.array(arr2_ret)

    def calc_kling_gupta_efficiency(self, observed, simulated):
        alpha = np.nanstd(simulated) / np.nanstd(observed)
        beta = np.nanmean(simulated) / np.nanmean(observed)
        r = np.corrcoef(observed, simulated)[1, 0]
        self.objectives.kge = 1 - np.sqrt(
            (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
        )
        self.objectives.alpha = alpha
        self.objectives.beta = beta
        self.objectives.r = r

    def calc_nash_sutcliff_efficiency(self, observed, simulated):
        self.objectives.nse = 1 - (
            np.nansum((observed - simulated) ** 2)
            / np.nansum((observed - np.mean(observed)) ** 2)
        )

    def calc_objectives(self, observed, simulated):
        observed, simulated = self.remove_empty_values(observed, simulated)
        self.calc_nash_sutcliff_efficiency(observed, simulated)
        self.calc_kling_gupta_efficiency(observed, simulated)
        self.objectives.diff = np.sum(simulated) - np.sum(observed)
        self.objectives.rel_diff = self.objectives.diff / np.sum(observed)

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
        self.logger.debug(f"{n} is last plot {np.sum(self.plots[n:]) == 1}")
        return np.sum(self.plots[n:]) == 1

    def get_catchment_area(self, path, ndecimal=0):
        config_file = Path(path + "ConfigFile.log")
        if config_file.exists():
            doc = config_file.open("r").readlines()
            for line in doc[::-1]:
                if line.strip():
                    self.catchment.area = (
                        f"{float(line.replace('Total[km2]', '').strip()):.{ndecimal}f}"
                    )
        self.logger.warning("Area could not be read.")

    @staticmethod
    def get_long_time_monthly_mean(variable):
        """Takes a variable and calculated the long time average value for every month of the year
        :param variable: xarray with a variable and a time
        :return: list of twelve numbers corresponding to the long term average value for each month of the year
        """
        var_ses = [[], [], [], [], [], [], [], [], [], [], [], []]
        for i in range(len(variable)):
            var_ses[int(variable.time[i].dt.month.data) - 1].append(variable[i])
        return np.array([np.mean(np.array(m)) for m in var_ses])

    @staticmethod
    def plot_on_axis(
        function, xvalues, yvalues: list, colors=None, labels=None, **arguments
    ):
        """Plots multiple graphs for specified function.
        :param function: matplotlib plot function e.g. ax.plot, plt.plot, ax.scatter, ax.errorbar, ...
        :param xvalues: list of x values
        :param yvalues: list of arrays with the y values
        :param colors: optional list of colors (default red, blue)
        :param labels: optional list of labels  (default simulated discharge, observed discharge)
        :param arguments: other arguments relevant for the given input function. e.g. linewidth=0.5 for 'plt.plot'
        """
        if labels is None:
            labels = ["simulated discharge", "observed discharge"]
        if colors is None:
            colors = ["#EF4340", "#3A4F99"]
        for i, yvalue in enumerate(yvalues):
            arguments["color"] = colors[i]
            arguments["label"] = labels[i]
            function(xvalues, yvalue, **arguments)

    def check_which_plots_to_create(self, code):
        """Determines which plots to create based on the given code.
        the produced tuple as a one at the index of the selected plots:
          model timestep (0), yearly (1), seasonality (2), scatter (3)
        """
        if not code:
            self.logger.warning("No plots will be produced since none were specified.")
        if "t" in code:
            self.plots[0] = 1
        if "y" in code:
            self.plots[1] = 1
        if "s" in code:
            self.plots[2] = 1
        if "c" in code:
            self.plots[3] = 1

    @staticmethod
    def raise_if_not_directory(path):
        p = Path(path)
        if not p.is_dir():
            msg = 'The given path "{path}" is not a directory.'
            raise NotADirectoryError(msg)

    def load_data_from_path(self, path):
        self.raise_if_not_directory(path)
        with xr.open_dataset(path + "discharge.nc") as ds:
            self.discharge_data = ds.load()
            for v in self.discharge_data.variables:
                if not isinstance(v, str):
                    msg = f"variable name is not a string - {v} - {type(v)}"
                    raise TypeError(msg)
                for key in ["sim", "obs"]:
                    if key in v:
                        self.catchment.name = str(int(v.split("_")[1]))
                        self.discharge_data = self.discharge_data.rename({v: key})

    def gen_hydrograph(self, input_path, output_file, show, save, title, plot_code):
        """Read in discharge data and plot the simulated against the observed discharge
        for different time resolutions and a seasonality as well as plotting simulated against observed discharge.
        :param input_path: Path to discharge.nc file
        :param output_file: Filename of the resulting file. e.g. hydrograph.png
        :param show: bool if plots should be shown or not
        :param save: bool if plots should be saved or not
        :param title: title given to the hydrograph
        :param plot_code: code indicating which plots to create
        """
        self.check_which_plots_to_create(plot_code)
        if sum(self.plots) == 0:
            self.logger.warning("Create no plots")
            return

        # load data
        if input_path[-1] != "/":
            input_path += "/"
        self.load_data_from_path(input_path)

        # calculate metrics at timestep resolution (generally hourly)
        self.calc_objectives(self.discharge_data["sim"], self.discharge_data["obs"])

        # create figure and determining the number of rows and cols
        fig = plt.figure(figsize=(7, 8))
        nrows = sum(self.plots) // 2 + 1
        ncols = 2
        self.logger.debug(f"nrows = {nrows} and ncols = {ncols}")

        # generate a grid indicating used and unused cells
        self.grid = [[False] * ncols for _ in range(nrows)]
        gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1])

        # write title
        self.get_catchment_area(input_path, ndecimal=0)
        fig.text(
            s=f"{self.catchment.name}",
            x=0.01,
            y=0.97,
            horizontalalignment="left",
            fontsize="x-large",
        )
        if self.catchment.area:
            fig.text(
                s="Area = " + self.catchment.area + r"$km^2$",
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

        # generate plots
        if self.plots[0]:
            self.logger.info("generating discharge plot")
            r, c = self.get_row_col()
            ax1 = fig.add_subplot(gs[r, c:])
            self.grid[r] = [True for _ in self.grid[r]]
            self.logger.debug(self.grid)
            self.plot_on_axis(
                function=ax1.scatter,
                xvalues=self.discharge_data["time"],
                yvalues=[self.discharge_data["sim"], self.discharge_data["obs"]],
                s=1.0,
            )
            self.plot_on_axis(
                function=ax1.plot,
                xvalues=self.discharge_data["time"],
                yvalues=[self.discharge_data["sim"], self.discharge_data["obs"]],
                linewidth=0.3,
            )
            ax1.legend()
            title = (
                f"NSE = {self.objectives.nse:.2f}, "
                f"KGE = {self.objectives.kge:.2f}, "
                f"alpha = {self.objectives.alpha:.2f}, "
                f"beta = {self.objectives.beta:.2f}"
            )
            if not self.plots[3]:
                title += f", r = {self.objectives.r:.2f}"
            ax1.set_title(
                title,
                horizontalalignment="center",
            )
            ax1.set_ylabel(r"Q $[m^3 s^{-1}]$")
            ax1.set_xlim(
                self.discharge_data["time"][0], self.discharge_data["time"][-1]
            )

        if self.plots[1]:
            self.logger.info("generating yearly discharge plot")
            r, c = self.get_row_col()
            self.logger.debug(f"yearly plot as row {r} and col {c}")
            if r == 0 or self.is_last_plot(1):
                ax3 = fig.add_subplot(gs[r, c:])
                self.grid[r] = [True for _ in self.grid[r]]
            else:
                ax3 = fig.add_subplot(gs[r, c])
                self.grid[r][c] = True
            # ax32 = ax3.twinx()
            self.logger.debug(self.grid)
            discharge_yearly = self.discharge_data.resample(time="Y").mean(skipna=False)
            # calculate metrics at yearly resolution
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
            # self.plot_on_axis(
            #     function=ax32.plot,
            #     xvalues=time_yearly,
            #     yvalues=[discharge_yearly["sim"]-discharge_yearly["obs"]],
            #     color=['black'],
            #     linewidth=0.3,
            # )
            if r == 0:
                ax3.legend()
            ax3.set_title(
                f"sim - obs = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                horizontalalignment="center",
            )
            # ax32.set_ylabel(r"difference $[m^3 s^{-1}]$")
            ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")
            ax3.set_xlim(time_yearly[0], time_yearly[-1])
            ax3.set_xticks(time_yearly[:: len(time_yearly) // 3])

        if self.plots[2]:
            self.logger.info("generating discharge seasonality plot")
            r, c = self.get_row_col()
            if r == 0 or self.is_last_plot(2):
                ax4 = fig.add_subplot(gs[r, c:])
                self.grid[r] = [True for _ in self.grid[r]]
            else:
                ax4 = fig.add_subplot(gs[r, c])
                self.grid[r][c] = True
            self.logger.debug(self.grid)
            self.plot_on_axis(
                function=ax4.scatter,
                xvalues=range(1, 13),
                yvalues=[
                    self.get_long_time_monthly_mean(self.discharge_data["sim"]),
                    self.get_long_time_monthly_mean(self.discharge_data["obs"]),
                ],
                s=1,
            )
            self.plot_on_axis(
                function=ax4.plot,
                xvalues=np.arange(1, 13),
                yvalues=[
                    self.get_long_time_monthly_mean(self.discharge_data["sim"]),
                    self.get_long_time_monthly_mean(self.discharge_data["obs"]),
                ],
                linewidth=0.3,
            )
            if r == 0:
                ax4.legend()
            ax4.set_title("Seasonality", horizontalalignment="center")
            ax4.set_xlim(0.9, 12.1)
            ax4.set_ylabel(r"Q $[m^3 s^{-1}]$")
            ax4.set_xlabel(r"month")
        if self.plots[3]:
            self.logger.info("generating discharge scatter plot")
            r, c = self.get_row_col()
            if r == 0 or self.is_last_plot(3):
                self.logger.debug("scatter is last")
                ax5 = fig.add_subplot(gs[r, c:])
                self.grid[r] = [True for _ in self.grid[r][c:]]
            else:
                ax5 = fig.add_subplot(gs[r, c])
                self.grid[r][c] = True
            self.logger.debug(self.grid)
            # add linear regression line to scatterplot
            self.plot_on_axis(
                function=ax5.scatter,
                xvalues=self.discharge_data["obs"],
                yvalues=[self.discharge_data["sim"]],
                s=50.0,
                colors=["black"],
                edgecolor="white",
                linewidth=0.3,
                alpha=0.1,
            )
            xvalues = np.linspace(0, 1e6, 10000)
            self.plot_on_axis(
                function=ax5.plot,
                xvalues=xvalues,
                yvalues=[
                    self.objectives.r * xvalues,
                    xvalues,
                ],
                colors=["red", "black"],
                linewidth=0.5,
            )
            if r == 0:
                ax5.legend()
            ax5.set_title(
                f"correlation coeff r = {self.objectives.r:.2f}",
                horizontalalignment="center",
            )
            ax5.set_xlabel("Qobs $[m^3 s^{-1}]$")  # X Achsenbeschriftung
            ax5.set_ylabel("Qsim $[m^3 s^{-1}]$")  # Y Achsenbeschriftung
            lim = (
                np.max(
                    [
                        np.max(self.discharge_data["obs"]),
                        np.max(self.discharge_data["sim"]),
                    ]
                )
                * 1.1
            )
            lim = lim - lim % 5 if lim - lim % 5 >= lim / 1.1 else lim + 5 - lim % 5
            ax5.set_xlim(0, lim)
            ax5.set_ylim(0, lim)
        plt.tight_layout()
        if save:
            if (
                len(output_file.split("/")) == 1
            ):  # by default the hydrograph is saved to the data directory
                output_file = input_path + output_file
            fig.savefig(output_file, bbox_inches="tight")
            self.logger.info(f"saved hydrograph to '{output_file}'")
        if show:
            plt.show()
        else:
            plt.close()
