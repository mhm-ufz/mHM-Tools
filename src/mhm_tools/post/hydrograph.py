"""
Plot a hydrograph with different time resolutions and a seasonality as well as plotting simulated against observed discharge.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec

from mhm_tools.common.logger import ErrorLogger, log_arguments

logger = logging.getLogger(__name__)


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
    gamma = None
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
        __init__(self): Initializes the Hydrograph object with the specified log level.
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
        load_data_from_discharge_nc(self, path): Loads discharge data from the specified path.
        load_precipiation_data(self, path): Loads precipitation data from the specified path.
        gen_hydrograph(self, input_path, output_file, show, save, title, plot_code, prec_path): Generates a hydrograph plot based on the specified parameters.
    """

    grid = None
    logging.basicConfig(format="%(asctime)s - %(levelname)-8s - %(message)s")
    logger = logging.getLogger(__name__)
    catchment = Catchment()
    sim_discharge_data = None
    obs_discharge_data = None
    sim_discharge_data_clean = None
    obs_discharge_data_clean = None
    sim_discharge_data_nonan = None
    obs_discharge_data_nonan = None
    pre = None
    objectives = Objectives()
    title = None
    output_file = None
    show = False
    save = False
    calc_stats = False

    def __init__(self, simulation=None, observation=None, calc_stats=True):
        self.plots = [0, 0, 0, 0]
        self.calc_stats = calc_stats
        if simulation is not None and observation is not None:
            self.set_discharge(simulation=simulation, observation=observation)

    def set_discharge(self, simulation=None, observation=None):
        """Set the discharge variables and remove nan values."""
        if simulation is not None and observation is not None:
            self.sim_discharge_data = simulation
            self.obs_discharge_data = observation
        elif simulation is not None or observation is not None:
            msg = "Either one or none of the input must be via array."
            raise ValueError(msg)
        logger.debug(f'Simulation input data: {simulation}')
        logger.debug(f'Observation input data: {observation}')
        self.sim_discharge_data_nonan = self.sim_discharge_data.dropna(
            dim="time", how="all"
        )
        self.obs_discharge_data_nonan = self.obs_discharge_data.dropna(
            dim="time", how="all"
        )
        if (
            self.sim_discharge_data_nonan.time.size == 0
            or self.obs_discharge_data_nonan.time.size == 0
        ):
            return False
        if simulation is not None and observation is not None and self.calc_stats:
            self.sim_discharge_data_clean, self.obs_discharge_data_clean = (
                self.remove_empty_values(
                    self.sim_discharge_data, self.obs_discharge_data
                )
            )
        return True

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
            exeption = f"The two timeseries do not have the same length. {arr1.shape} and {arr2.shape}"
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
        gamma = np.corrcoef(observed, simulated)[1, 0]
        self.objectives.kge = 1 - np.sqrt(
            (gamma - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
        )
        self.objectives.alpha = alpha
        self.objectives.beta = beta
        self.objectives.gamma = gamma

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
            / np.nansum((observed - np.nanmean(observed)) ** 2)
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
        if (
            self.obs_discharge_data_clean is None
            or self.sim_discharge_data_clean is None
        ):
            self.obs_discharge_data_clean, self.sim_discharge_data_clean = (
                self.remove_empty_values(observed, simulated)
            )
        if np.all(np.isnan(self.obs_discharge_data_clean)) or np.all(
            np.isnan(self.sim_discharge_data_clean)
        ):
            logger.warning(
                f"In calc objectives: obs is nan {np.all(np.isnan(self.obs_discharge_data_clean))} or sim is nan {np.all(np.isnan(self.sim_discharge_data_clean))}"
            )
            return False
        self.calc_nash_sutcliff_efficiency(
            self.obs_discharge_data_clean, self.sim_discharge_data_clean
        )
        self.calc_kling_gupta_efficiency(
            self.obs_discharge_data_clean, self.sim_discharge_data_clean
        )
        self.logger.debug(
            f"sum simulated: {np.nansum(self.sim_discharge_data_clean)}, sum observed: {np.nansum(self.obs_discharge_data_clean)}"
        )
        self.objectives.diff = np.nansum(self.sim_discharge_data_clean) - np.nansum(
            self.obs_discharge_data_clean
        )
        self.objectives.rel_diff = self.objectives.diff / np.nansum(
            self.obs_discharge_data_clean
        )
        return True

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
        config_file = Path(path / "ConfigFile.log")
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
    def get_long_time_monthly_mean(variable, long=False):
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
        var_ses = [np.nanmean(var_ses[i]) for i in range(12)]
        if long:
            var_ses = [var_ses[-1], *var_ses, var_ses[0]]
        return np.array(var_ses)

    @staticmethod
    def plot_on_axis(
        function, yvalues: list, xvalues=None, colors=None, labels=None, **arguments
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
            if xvalues is None:
                function(yvalue.time, yvalue, **arguments)
            elif len(xvalues) == len(yvalues):
                function(xvalues[i], yvalue, **arguments)
            else:
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
        if "c" in code and self.calc_stats:
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

    def load_data_from_discharge_nc(self, path):
        """
        Load discharge data from the specified path.

        Args:
            path (str): The path to the directory containing the discharge data.

        Raises
        ------
            TypeError: If the variable name in the discharge dataset is not a string.

        """
        path = Path(path)
        discharge_file = path / "discharge.nc"
        if discharge_file.is_file():
            with xr.open_dataset(path / "discharge.nc") as ds:
                discharge_data = ds.load()
                for v in discharge_data.variables:
                    if not isinstance(v, str):
                        msg = f"variable name is not a string - {v} - {type(v)}"
                        raise TypeError(msg)
                    if "sim" in v:
                        self.catchment.name = str(int(v.split("_")[1]))
                        self.sim_discharge_data = discharge_data[v]
                    if "obs" in v:
                        self.catchment.name = str(int(v.split("_")[1]))
                        self.obs_discharge_data = discharge_data[v]
            self.set_discharge()
            return True
        return False

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
            return
        path = Path(path)
        if path.is_dir():
            if (path / "pre.nc").is_file():
                path = path / "pre.nc"
            elif (path / "pre" / "pre.nc").is_file():
                path = path / "pre" / "pre.nc"
            else:
                msg = f"no precipitation file found in the directory {path}"
                self.logger.warning(msg)
                return
        elif not path.is_file:
            msg = f"{path} is neither a directory nor a file"
            self.logger.warning(msg)
            return
        with xr.open_dataset(path) as ds:
            self.pre = ds.load()

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
        # self.plot_on_axis(
        #     function=ax1.scatter,
        #     xvalues=self.sim_discharge_data["time"],
        #     yvalues=[self.sim_discharge_data, self.obs_discharge_data],
        #     s=1.0,
        # )
        self.plot_on_axis(
            function=ax1.plot,
            yvalues=[self.sim_discharge_data, self.obs_discharge_data],
            linewidth=0.3,
            # labels=[],  # no labels
        )
        ax1.legend()
        title = ""
        if self.calc_stats:
            title = (
                f"NSE = {self.objectives.nse:.2f}, "
                f"KGE = {self.objectives.kge:.2f}, "
                f"alpha = {self.objectives.alpha:.2f}, "
                f"beta = {self.objectives.beta:.2f}"
            )
            if not self.plots[3]:
                title += f", r = {self.objectives.gamma:.2f}"
        ax1.set_title(
            title,
            horizontalalignment="center",
        )
        ax1.set_ylabel(r"Q $[m^3 s^{-1}]$")
        xmin = min(
            self.sim_discharge_data_nonan.time.min(),
            self.obs_discharge_data_nonan.time.min(),
        )
        xmax = max(
            self.sim_discharge_data_nonan.time.max(),
            self.obs_discharge_data_nonan.time.max(),
        )
        ax1.set_xlim(xmin, xmax)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
        )

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
        # calculate metrics at yearly resolution
        if np.all(np.isnan(self.sim_discharge_data)) or np.all(
            np.isnan(self.obs_discharge_data)
        ):
            logger.warning(
                "Cannot create yearly plot because one of the dataarrays is empty except for nan values."
            )
            return
        sim_discharge_yearly = self.sim_discharge_data_nonan.resample(time="YE").mean(
            skipna=True
        )
        obs_discharge_yearly = self.obs_discharge_data_nonan.resample(time="YE").mean(
            skipna=True
        )
        time_yearly_sim = [int(y.dt.year.data) for y in sim_discharge_yearly.time]
        time_yearly_obs = [int(y.dt.year.data) for y in obs_discharge_yearly.time]
        years_combined = np.unique(time_yearly_sim + time_yearly_obs)
        years_combined.sort()
        if years_combined is None or len(years_combined) < 3:
            logger.warning(
                "Cannot create yearly plot because the data is insufficient."
            )
            return
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

        if pre is not None:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=outer_gs, height_ratios=[1, 1, 1]
            )
            ax2_pre = fig.add_subplot(inner_gs[0])
            # ax2_pre.spines["top"].set_visible(False)
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
            ax2_pre.set_ylim(-np.nanmax(pre_yearly) - np.nanmax(pre_yearly) / 6, 0)
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
            if self.calc_stats:
                ax2_pre.set_title(
                    f"sim - obs = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                    horizontalalignment="center",
                )
        else:
            ax2 = fig.add_subplot(outer_gs)
            if self.calc_stats:
                ax2.set_title(
                    f"sim - obs = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                    horizontalalignment="center",
                )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        self.plot_on_axis(
            function=ax2.scatter,
            xvalues=[time_yearly_sim, time_yearly_obs],
            yvalues=[sim_discharge_yearly, obs_discharge_yearly],
            s=1.0,
        )
        self.plot_on_axis(
            function=ax2.plot,
            xvalues=[time_yearly_sim, time_yearly_obs],
            yvalues=[sim_discharge_yearly, obs_discharge_yearly],
            linewidth=0.3,
        )
        if r == 0:
            ax2.legend()
        ax2.set_ylabel(r"Q $[m^3 s^{-1}]$")
        ax2.set_xlim(np.min(years_combined) - 0.5, np.max(years_combined) + 0.5)
        ax2.set_xticks(
            years_combined[:: len((years_combined) - np.min(years_combined)) // 3]
        )

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
            # ax3_pre.spines["top"].set_visible(False)
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
            ax3_pre.set_ylim(-np.nanmax(pre_monthly) - np.nanmax(pre_monthly) / 6, 0)
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

        else:
            ax3 = fig.add_subplot(outer_gs)
            ax3.set_title("Seasonality", horizontalalignment="center")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        season_sim = self.get_long_time_monthly_mean(self.sim_discharge_data, long=True)
        season_obs = self.get_long_time_monthly_mean(self.obs_discharge_data, long=True)
        self.logger.debug(f"sim: {season_sim}")
        self.logger.debug(f"osb: {season_obs}")
        self.plot_on_axis(
            function=ax3.scatter,
            xvalues=np.arange(0, 14),
            yvalues=[
                season_sim,
                season_obs,
            ],
            s=1,
        )
        self.plot_on_axis(
            function=ax3.plot,
            xvalues=np.arange(0, 14),
            yvalues=[season_sim, season_obs],
            linewidth=0.3,
        )

        if r == 0:
            ax3.legend()

        ax3.set_xlim(0.5, 12.5)
        ax3.set_xticks(
            np.arange(1, 13),
        )
        ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")

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
            xvalues=self.obs_discharge_data_clean,
            yvalues=[self.sim_discharge_data_clean],
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
        if self.calc_stats:
            ax4.set_title(
                f"correlation coeff r = {self.objectives.gamma:.2f}",
                horizontalalignment="center",
            )
        ax4.set_xlabel("observed $[m^3 s^{-1}]$")  # X Achsenbeschriftung
        ax4.set_ylabel("simulated $[m^3 s^{-1}]$")  # Y Achsenbeschriftung
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        lim = (
            np.nanmax(
                [
                    np.nanmax(self.obs_discharge_data),
                    np.nanmax(self.sim_discharge_data),
                ]
            )
            * 1.1
        )
        lim = lim - lim % 5 if lim - lim % 5 >= lim / 1.1 else lim + 5 - lim % 5
        ax4.set_xlim(0, lim)
        ax4.set_ylim(0, lim)

    def crop_data_to_overlapping_time(self):
        """Crop data to overlapping time."""
        t1 = self.sim_discharge_data.dropna(dim="time", how="all").time.values
        t2 = self.obs_discharge_data.dropna(dim="time", how="all").time.values

        # Find overlapping range
        only_nan_msg = "No non nan value data."
        if t1.any() and t2.any():
            start = max(t1[0], t2[0])
            end = min(t1[-1], t2[-1])
            if end <= start:
                logger.warning(
                    f"The two datasets are not overlapping. Sim data hass non nan data from {t1[0]} to {t1[-1]} and obs from {t2[0]} to {t2[-1]}."
                )
                return False
            logger.info(f"Cropping data to timeframe {start} to {end}")
            # Slice both datasets to that time range
            self.sim_discharge_data = self.sim_discharge_data.sel(
                time=slice(start, end)
            )
            self.obs_discharge_data = self.obs_discharge_data.sel(
                time=slice(start, end)
            )
            if np.all(np.isnan(self.sim_discharge_data)) or np.all(
                np.isnan(self.sim_discharge_data)
            ):
                logger.warning(
                    f"In crop data: obs is nan {not t1} or sim is nan {not t2}"
                )
                raise ValueError(only_nan_msg)
        else:
            logger.warning(
                f"In crop data: obs is nan {t1.any()} or sim is nan {t2.any() }"
            )
            raise ValueError(only_nan_msg)
        logger.info(f"sim_discharge_data: {self.sim_discharge_data.data}")
        logger.info(f"obs_discharge_data: {self.obs_discharge_data.data}")
        if self.pre is not None:
            self.pre = self.pre.sel(time=slice(start, end))
        return True

    def get_hydrograph(self):
        """Generate the hydrograph from the data saved as member variables."""
        if sum(self.plots) == 0:
            self.logger.warning("Create no plots")
            return None
        # load data

        # calculate metrics at timestep resolution (generally hourly)
        self.logger.debug(self.sim_discharge_data)
        if self.calc_stats:
            logger.info("Crop to overlapping time.")
            if not self.crop_data_to_overlapping_time():
                return False
            logger.info("Calculate objectives")
            if not self.calc_objectives(
                self.obs_discharge_data, self.sim_discharge_data
            ):
                return False

        # create figure and determining the number of rows and cols
        fig = plt.figure(figsize=(7, 8))
        nrows = sum(self.plots) // 2 + 1
        ncols = 2
        self.logger.debug(f"nrows = {nrows} and ncols = {ncols}")

        # generate a grid indicating used and unused cells
        self.grid = [[False] * ncols for _ in range(nrows)]
        gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1])

        # write title
        fig.text(
            s=f"{self.catchment.name}",
            x=0.01,
            y=0.97,
            horizontalalignment="left",
            fontsize="x-large",
        )
        if self.catchment.area:
            fig.text(
                s=f"Area = {self.catchment.area:.2f}" + r"$km^2$",
                x=0.5,
                y=0.97,
                horizontalalignment="center",
                fontsize="x-large",
            )
        fig.suptitle(
            t=f"\n{self.title}\n",
            x=0.5,
            y=0.97,
            horizontalalignment="center",
            fontsize="x-large",
        )

        # generateself plots
        if self.plots[0]:
            self.create_plot_at_timestep(fig, gs)
        if self.plots[1]:
            self.create_plot_yearly(fig, gs, self.pre)
        if self.plots[2]:
            self.create_plot_seasonality(fig, gs, self.pre)
        if self.plots[3]:
            self.create_plot_scatter(fig, gs)
        plt.tight_layout()
        if self.save:
            # if (
            #     len(self.output_file.split("/")) == 1
            # ):  # by default the hydrograph is saved to the data directory
            # self.output_file = self.output_file
            fig.savefig(self.output_file, bbox_inches="tight")
            self.logger.info(f"saved hydrograph to '{self.output_file}'")
        if self.show:
            plt.show()
        else:
            plt.close()
        return True


@log_arguments()
def get_hydrograph_from_path(
    input_path, output_file, show, save, title, plot_code, prec_path
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
    hydro = Hydrograph()
    hydro.check_which_plots_to_create(plot_code)
    hydro.output_file = output_file
    hydro.title = title
    hydro.show = show
    hydro.save = save
    input_path = Path(input_path)
    prec_path = Path(prec_path)
    if not hydro.load_data_from_discharge_nc(input_path):
        msg = f"No discharge.nc file at {input_path}"
        with ErrorLogger():
            raise ValueError(msg)
    hydro.load_precipiation_data(prec_path)
    hydro.get_catchment_area(input_path, ndecimal=0)
    hydro.get_hydrograph()


@log_arguments()
def gen_hydrograph_by_data_sets(
    simulations,
    observation,
    precipitation,
    output_file,
    area=None,
    plot_code="tysc",
    title="",
    show=False,
    save=True,
    id=None,
    calc_stats=False,
    raise_exceptions=True
):
    """
    Use discharge and precipitation data provided as xarrays to produce a hydrograph with different analysises.

    Simulated and observed discharge are plotted for different temporal resolutions.
    Additionally a seasonality as well as a scatter-plot simulated against observed discharge are produced.

    Args:
        simulations: xarray or list of xarrays with simulation data
        observation: xarray with observation data
        precipitation: xarray with precipiation data
        input_path: Path to mhm output
        output_file: Filename of the resulting file. e.g. hydrograph.png
        show: bool if plots should be shown or not
        save: bool if plots should be saved or not
        title: title given to the hydrograph
        raise_exceptions: Raise or only log exceptions
        plot_code: code indicating which plots to create
    """
    try:
        hydro = Hydrograph(calc_stats=calc_stats)
        missing_data_error_msg = f"For {id} the hydrograph could not be created."
        if hydro.set_discharge(simulation=simulations, observation=observation):
            hydro.pre = precipitation
            hydro.output_file = output_file
            hydro.title = str(id) if not title and id is not None else title
            hydro.show = show
            hydro.save = save
            hydro.catchment.area = area
            hydro.check_which_plots_to_create(plot_code)
            if not hydro.get_hydrograph():
                logger.error(missing_data_error_msg)
        else:
            logger.error(missing_data_error_msg)
        return {**hydro.objectives.__dict__, "id": id}
    except Exception as e:
        logger.error(e)
        if raise_exceptions:
            raise e
    return {"id": id}