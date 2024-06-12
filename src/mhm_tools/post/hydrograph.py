"""
Plot a hydrograph with different time resolutions and a seasonality as well as plotting simulated against observed discharge.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec


class Catchment:
    """
    Represents a catchment.

    Attributes
    ----------
        name (str): The name of the catchment.
        area (float): The area of the catchment in square units.
    """

    name = None
    area = None


class Objectives:
    """
    A class representing the objectives for evaluating river discharge data.

    Attributes
    ----------
        kge (float): Kling-Gupta Efficiency.
        nse (float): Nash-Sutcliffe Efficiency.
        alpha (float): Alpha objective.
        beta (float): Beta objective.
        r (float): Pearson correlation coefficient.
        diff (float): Difference objective.
        rel_diff (float): Relative difference objective.
    """

    kge = None
    nse = None
    alpha = None
    beta = None
    r = None
    diff = None
    rel_diff = None


class Hydrograph:
    """
    Represents a hydrograph and provides methods for calculating metrics and generating plots.

    Attributes
    ----------
        levels (dict): A dictionary mapping log levels to their corresponding values.
        grid (list): A 2D list representing a grid of used and unused cells for plots.
        logger (logging.Logger): A logger object for logging messages.
        plots (list): A list indicating which plots to create.
        catchment (Catchment): An instance of the Catchment class.
        discharge_data (xarray.Dataset): A dataset containing discharge data.
        objectives (Objectives): An instance of the Objectives class.

    Methods
    -------
        __init__(self, log_level): Initializes the Hydrograph object with the specified log level.
        remove_empty_values(arr1, arr2): Removes empty values from two arrays and returns the cleaned data.
        calc_kling_gupta_efficiency(observed, simulated): Calculates the Kling-Gupta efficiency metric.
        calc_nash_sutcliff_efficiency(observed, simulated): Calculates the Nash-Sutcliffe efficiency metric.
        calc_objectives(observed, simulated): Calculates various metrics based on observed and simulated data.
        get_row_col(self): Finds the first unused gridcell for the next plot.
        is_last_plot(self, n): Checks if the given plot is the last one.
        get_catchment_area(self, path, ndecimal=0): Retrieves the catchment area from a config file.
        get_long_time_monthly_mean(variable): Calculates the long-term average value for each month of the year.
        plot_on_axis(function, xvalues, yvalues, colors=None, labels=None, **arguments): Plots multiple graphs using the specified function.
        check_which_plots_to_create(self, code): Determines which plots to create based on the given code.
        raise_if_not_directory(path): Raises an exception if the given path is not a directory.
        load_data_from_path(self, path): Loads discharge data from the specified path.
        load_precipiation_data(self, path): Loads precipitation data from the specified path.
        gen_hydrograph(self, input_path, output_file, show, save, title, plot_code, prec_path): Generates a hydrograph plot based on the specified parameters.
    """

    grid = None
    logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
    logger = logging.getLogger(__name__)
    catchment = Catchment()
    discharge_data = None
    objectives = Objectives()

    def __init__(self, log_level):
        levels = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "debug": logging.DEBUG,
            "error": logging.ERROR,
        }
        self.logger.setLevel(levels[log_level])
        self.plots = [0, 0, 0, 0]

    def remove_empty_values(self, arr1, arr2, recursive=True):
        """
        Remove empty values from two arrays and return the cleaned data.

        Args:
            arr1 : list
                The first array.
            arr2 : list
                The second array.

        Returns
        -------
            tuple
                A tuple containing the cleaned data arrays in the order they were given as arguments.
        """
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        if len(arr1) != len(arr2):
            exeption = "The two timeseries do not have the same length."
            raise ValueError(exeption)
        data = np.transpose(np.array([arr1.flatten(), arr2.flatten()]))
        try:
            data = data[~np.isnan(data).any(1)]
            return data[:, 0], data[:, 1]
        except TypeError as te:
            if recursive:
                return self.remove_empty_values(
                    [v if v is not None else np.nan for v in arr1],
                    [v if v is not None else np.nan for v in arr2],
                    recursive=False,
                )
            msg = "While removing the empty values a wrong input type was detected."
            raise TypeError(msg) from te
        # arr1_ret, arr2_ret = [], []
        # for i, v in enumerate(arr1):
        #     w = arr2[i]
        #     if v is not None and v is not np.nan and w is not None and w is not np.nan:
        #         arr1_ret.append(v)
        #         arr2_ret.append(w)
        # return np.array(arr1_ret), np.array(arr2_ret)

    def calc_kling_gupta_efficiency(self, observed, simulated):
        """
        Calculate the kling-gupta efficiency metric and saves it as well as its components in the objectives.

        Args:
            observed : list
                The observed discharge data.
            simulated : list
                The simulated discharge data.

        Returns
        -------
            None
        """
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
        """
        Calculate the Nash-Sutcliffe efficiency metric and save it in the objectives.

        Args:
            observed : list
                The observed discharge data.
            simulated : list
                The simulated discharge data.

        Returns
        -------
            None
        """
        self.objectives.nse = 1 - (
            np.nansum((observed - simulated) ** 2)
            / np.nansum((observed - np.mean(observed)) ** 2)
        )

    def calc_objectives(self, observed, simulated):
        """
        Remove empty values from the data arrays and calculate the objectives.

        Args:
        observed : list
            The observed discharge data.
        simulated : list
            The simulated discharge data.

        Returns
        -------
        None
        """
        observed, simulated = self.remove_empty_values(observed, simulated)
        self.calc_nash_sutcliff_efficiency(observed, simulated)
        self.calc_kling_gupta_efficiency(observed, simulated)
        self.logger.debug(
            f"sum simulated: {np.nansum(simulated)}, sum observed: {np.nansum(observed)}"
        )
        self.objectives.diff = np.nansum(simulated) - np.nansum(observed)
        self.objectives.rel_diff = self.objectives.diff / np.nansum(observed)

    def get_row_col(self):
        """
        Find the first unused gridcell for the next plot.

        Returns
        -------
            Tuple[int, int]: The row and column indices of the first unused gridcell.

        Raises
        ------
            ValueError: If there are no unused gridcells left.
        """
        for i, row in enumerate(self.grid):
            for j, v in enumerate(row):
                if not v:
                    return i, j
        msg = "No unused gridcell left"
        raise ValueError(msg)

    def is_last_plot(self, n):
        """
        Check if the given plot index is the last plot.

        Args:
        - n (int): The plot index to check.

        Returns
        -------
        - bool: True if the given plot index is the last plot, False otherwise.
        """
        # finds the first unused gridcell for the next plot
        self.logger.debug(f"{n} is last plot {np.sum(self.plots[n:]) == 1}")
        return np.sum(self.plots[n:]) == 1

    def get_catchment_area(self, path, ndecimal=0):
        """
        Retrieve the catchment area from the given mHM ConfigFile.log.

        Args:
            path (str): The path to the file.
            ndecimal (int, optional): The number of decimal places to round the catchment area to. Defaults to 0.

        Returns
        -------
            None

        Raises
        ------
            None
        """
        config_file = Path(path + "ConfigFile.log")
        if config_file.exists():
            with config_file.open() as f:
                doc = f.readlines()
                for line in doc[::-1]:
                    if line.strip() and "Total[km2]" in line:
                        self.catchment.area = f"{float(line.replace('Total[km2]', '').strip()):.{ndecimal}f}"
                        self.logger.debug(f"Area: {self.catchment.area}")
                        return
        self.logger.warning("Area could not be read.")

    @staticmethod
    def get_long_time_monthly_mean(variable):
        """
        Calculate the long-term average value for every month of the year for a given variable.

        Args:
            variable: xarray with a variable and a time
        Returns:
            list of twelve numbers corresponding to the long-term average value for each month of the year
        """
        var_ses = [[], [], [], [], [], [], [], [], [], [], [], []]
        for i in range(len(variable)):
            var_ses[int(variable.time[i].dt.month.data) - 1].append(variable[i])
        return np.array([np.nanmean(np.array(m)) for m in var_ses])

    @staticmethod
    def plot_on_axis(
        function, xvalues, yvalues: list, colors=None, labels=None, **arguments
    ):
        """
        Plot multiple graphs for specified function.

        Args:
            function: matplotlib plot function e.g. ax.plot, plt.plot, ax.scatter, ax.errorbar, ...
            xvalues: list of x values
            yvalues: list of arrays with the y values
            colors: optional list of colors (default red, blue)
            labels: optional list of labels  (default simulated discharge, observed discharge)
            arguments: other arguments relevant for the given input function. e.g. linewidth=0.5 for 'plt.plot'
        """
        if labels is None:
            labels = ["simulated discharge", "observed discharge"]
        if colors is None:
            colors = ["#EF4340", "#3A4F99"]
        for i, yvalue in enumerate(yvalues):
            arguments["color"] = colors[i]
            if labels:
                arguments["label"] = labels[i]
            function(xvalues, yvalue, **arguments)

    def check_which_plots_to_create(self, code):
        """
        Determine which plots to create based on the given code.

        A tuple is produced and saved as member variable. It has a one at the index of the selected plots and a zero otherwise.
        The indices of the plots are:
        model timestep (0), yearly (1), seasonality (2), scatter (3)

        Args:
            code (str): A string indicating which plots to create.

        Returns
        -------
            None
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
        """
        Raise a NotADirectoryError if the given path is not a directory.

        Args:
            path (str): The path to check.

        Raises
        ------
            NotADirectoryError: If the given path is not a directory.
        """
        p = Path(path)
        if not p.is_dir():
            msg = f'The given path "{path}" is not a directory.'
            raise NotADirectoryError(msg)

    def load_data_from_path(self, path):
        """
        Load discharge data from the specified path.

        Args:
            path (str): The path to the directory containing the discharge data.

        Raises
        ------
            TypeError: If the variable name in the discharge dataset is not a string.

        """
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

    def load_precipiation_data(self, path):
        """
        Load precipitation data from a given file or directory path.

        Args:
            path (str): The path to the file or directory containing the precipitation data.

        Returns
        -------
            xr.Dataset: The loaded precipitation data as an xarray Dataset object.
                       Returns None if no precipitation file is found or if the path is invalid.
        """
        if path is None:
            return None
        path = Path(path)
        if path.is_dir():
            if (path / "pre.nc").is_file():
                path = path / "pre.nc"
            elif (path / "pre" / "pre.nc").is_file():
                path = path / "pre" / "pre.nc"
            else:
                msg = f"no precipitation file found in the directory {path}"
                self.logger.warning(msg)
                return None
        elif not path.is_file:
            msg = f"{path} is neither a directory nor a file"
            self.logger.warning(msg)
            return None
        with xr.open_dataset(path) as ds:
            return ds.load()

    def create_plot_at_timestep(self, fig, gs):
        """
        Create a discharge plot temporal resolution of the discharge output (daily or hourly).

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The gridspec object specifying the subplot layout.

        Returns
        -------
            None
        """
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
            labels=[],  # no labels
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
        ax1.set_xlim(self.discharge_data["time"][0], self.discharge_data["time"][-1])

    def create_plot_yearly(self, fig, gs, pre):
        """
        Generate a yearly discharge plot.

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The grid specification for the plot layout.
            pre (xarray.DataArray): The precipitation data.

        Returns
        -------
            None
        """
        self.logger.info("generating yearly discharge plot")
        r, c = self.get_row_col()
        self.logger.debug(f"yearly plot as row {r} and col {c}")
        if r == 0 or self.is_last_plot(1):
            outer_gs = gs[r, c:]
            self.grid[r] = [True for _ in self.grid[r]]
        else:
            outer_gs = gs[r, c]
            self.grid[r][c] = True
        self.logger.debug(self.grid)
        discharge_yearly = self.discharge_data.resample(time="YE").mean(skipna=True)

        if pre is not None:

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs, height_ratios=[1, 1, 1]
            )
            ax2_pre = fig.add_subplot(inner_gs[0])
            ax2_pre.spines["top"].set_visible(False)
            ax2_pre.spines["right"].set_visible(False)
            ax2_pre.spines["bottom"].set_visible(False)
            # plot precipitation as bar plot
            pre_yearly = pre["pre"].resample(time="YE").mean(skipna=True)
            time_yearly_pre = [int(y.dt.year.data) for y in pre_yearly["time"]]
            pre_yearly = np.nanmean(pre_yearly, axis=(1, 2))
            ax2_pre.bar(
                time_yearly_pre,
                -pre_yearly,
                color="darkblue",
                alpha=0.6,
                label="Precipitation",
            )
            ax2_pre.set_ylim(-np.max(pre_yearly) - np.max(pre_yearly) / 6, 0)
            ax2_pre.set_yticklabels(
                [f"{int(abs(tick))}" for tick in ax2_pre.get_yticks()]
            )
            ax2_pre.set_ylabel("pre [mm/day]", color="darkblue")
            ax2_pre.tick_params(axis="y", labelcolor="darkblue")
            ax2_pre.yaxis.set_label_position("right")
            ax2_pre.tick_params(
                top=False, labeltop=False, bottom=False, labelbottom=False
            )
            inner_gs_update = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs, height_ratios=[0.75, 1.25, 1]
            )
            ax2 = fig.add_subplot(inner_gs_update[1:], sharex=ax2_pre)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2_pre.set_title(
                f"sim - obs = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                horizontalalignment="center",
            )
        else:
            ax2 = fig.add_subplot(outer_gs)
            ax2.set_title(
                f"sim - obs = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                horizontalalignment="center",
            )
        # calculate metrics at yearly resolution
        time_yearly = [int(y.dt.year.data) for y in discharge_yearly["time"]]
        self.plot_on_axis(
            function=ax2.scatter,
            xvalues=time_yearly,
            yvalues=[discharge_yearly["sim"], discharge_yearly["obs"]],
            s=1.0,
        )
        self.plot_on_axis(
            function=ax2.plot,
            xvalues=time_yearly,
            yvalues=[discharge_yearly["sim"], discharge_yearly["obs"]],
            linewidth=0.3,
        )
        if r == 0:
            ax2.legend()
        ax2.set_ylabel(r"Q $[m^3 s^{-1}]$")
        ax2.set_xlim(time_yearly[0] - 0.5, time_yearly[-1] + 0.5)
        ax2.set_xticks(time_yearly[:: len(time_yearly) // 3])

    def create_plot_seasonality(self, fig, gs, pre):
        """
        Generate a discharge seasonality plot.

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The grid specification for the plot layout.
            pre (dict): The precipitation data.

        Returns
        -------
            None
        """
        self.logger.info("generating discharge seasonality plot")
        r, c = self.get_row_col()
        if r == 0 or self.is_last_plot(2):
            outer_gs = gs[r, c:]
            self.grid[r] = [True for _ in self.grid[r]]
        else:
            outer_gs = gs[r, c]
            self.grid[r][c] = True

        if pre is not None:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs, height_ratios=[1, 1, 1]
            )
            ax3_pre = fig.add_subplot(inner_gs[0])
            ax3_pre.set_title("Seasonality", horizontalalignment="center")
            ax3_pre.spines["top"].set_visible(False)
            ax3_pre.spines["right"].set_visible(False)
            ax3_pre.spines["bottom"].set_visible(False)
            # plot precipitation as bar plot
            pre_monthly = self.get_long_time_monthly_mean(pre["pre"])
            ax3_pre.bar(
                np.arange(1, 13),
                -pre_monthly,
                color="darkblue",
                alpha=0.6,
                label="Precipitation",
            )
            ax3_pre.set_ylim(-np.max(pre_monthly) - np.max(pre_monthly) / 6, 0)
            # ax3_pre.set_yticks([-100, -75, -50, -25, 0])  # Set specific tick positions for clarity
            ax3_pre.set_yticklabels(
                [f"{int(abs(tick))}" for tick in ax3_pre.get_yticks()]
            )
            ax3_pre.set_ylabel("pre [mm/day]", color="darkblue")
            ax3_pre.tick_params(axis="y", labelcolor="darkblue")
            # ax3_pre.set_xticks([])  # Hide x-ticks for the upper plot
            ax3_pre.yaxis.set_label_position("right")
            ax3_pre.tick_params(
                top=False, labeltop=False, bottom=False, labelbottom=False
            )
            inner_gs_update = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs, height_ratios=[0.75, 1.25, 1]
            )
            ax3 = fig.add_subplot(inner_gs_update[1:], sharex=ax3_pre)
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)

        else:
            ax3 = fig.add_subplot(outer_gs)
            ax3.set_title("Seasonality", horizontalalignment="center")
        season_sim = self.get_long_time_monthly_mean(self.discharge_data["sim"])
        season_obs = self.get_long_time_monthly_mean(self.discharge_data["obs"])
        self.logger.debug(f"sim: {season_sim}")
        self.logger.debug(f"osb: {season_obs}")
        self.plot_on_axis(
            function=ax3.scatter,
            xvalues=range(1, 13),
            yvalues=[
                season_sim,
                season_obs,
            ],
            s=1,
        )
        self.plot_on_axis(
            function=ax3.plot,
            xvalues=np.arange(1, 13),
            yvalues=[season_sim, season_obs],
            linewidth=0.3,
        )

        if r == 0:
            ax3.legend()

        ax3.set_xlim(0.7, 12.3)
        ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")
        ax3.set_xlabel(r"month")

    def create_plot_scatter(self, fig, gs):
        """
        Create a scatter plot, plotting the simulated against the observed discharge data.

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the scatter plot to.
            gs (matplotlib.gridspec.GridSpec): The gridspec object specifying the subplot layout.

        Returns
        -------
            None
        """
        self.logger.info("generating discharge scatter plot")
        r, c = self.get_row_col()
        if r == 0 or self.is_last_plot(3):
            self.logger.debug("scatter is last")
            ax4 = fig.add_subplot(gs[r, c:])
            self.grid[r] = [True for _ in self.grid[r][c:]]
        else:
            ax4 = fig.add_subplot(gs[r, c])
            self.grid[r][c] = True
        self.logger.debug(self.grid)
        # add linear regression line to scatterplot
        self.plot_on_axis(
            function=ax4.scatter,
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
            function=ax4.plot,
            xvalues=xvalues,
            yvalues=[
                # self.objectives.r * xvalues,
                xvalues,
            ],
            colors=[
                "red",
                # "black"
            ],
            linewidth=0.5,
        )
        if r == 0:
            ax4.legend()
        ax4.set_title(
            f"correlation coeff r = {self.objectives.r:.2f}",
            horizontalalignment="center",
        )
        ax4.set_xlabel("observed $[m^3 s^{-1}]$")  # X Achsenbeschriftung
        ax4.set_ylabel("simulated $[m^3 s^{-1}]$")  # Y Achsenbeschriftung
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
        ax4.set_xlim(0, lim)
        ax4.set_ylim(0, lim)

    def gen_hydrograph(
        self, input_path, output_file, show, save, title, plot_code, prec_path
    ):
        """
        Read in discharge data and produce a hydrograph with different analysises.

        Simulated and observed discharge are plotted for different temporal resolutions.
        Additionally a seasonality as well as a scatter-plot simulated against observed discharge are produced.

        Args:
            input_path: Path to discharge.nc file
            output_file: Filename of the resulting file. e.g. hydrograph.png
            show: bool if plots should be shown or not
            save: bool if plots should be saved or not
            title: title given to the hydrograph
            plot_code: code indicating which plots to create
        """
        self.check_which_plots_to_create(plot_code)
        if sum(self.plots) == 0:
            self.logger.warning("Create no plots")
            return

        # load data
        if input_path[-1] != "/":
            input_path += "/"
        self.load_data_from_path(input_path)
        pre = self.load_precipiation_data(prec_path)

        # calculate metrics at timestep resolution (generally hourly)
        self.logger.debug(self.discharge_data["sim"])
        self.calc_objectives(self.discharge_data["obs"], self.discharge_data["sim"])

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

        # generateself plots
        if self.plots[0]:
            self.create_plot_at_timestep(fig, gs)
        if self.plots[1]:
            self.create_plot_yearly(fig, gs, pre)
        if self.plots[2]:
            self.create_plot_seasonality(fig, gs, pre)
        if self.plots[3]:
            self.create_plot_scatter(fig, gs)
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
