"""Plot discharge diagnostics for one or more mHM/mRM simulations.

The module reads simulated and optional observed discharge, derives catchment
metadata, computes summary metrics, and creates time-series, yearly,
seasonality, flow-duration, and scatter hydrograph plots.

Authors
-------
- Simon Lüdke
"""

import logging
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import gridspec

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.metrics.metrics_handler import create_csv_from_dict
from mhm_tools.common.utils import dict_to_multiline_string

logger = logging.getLogger(__name__)


def _ensure_non_interactive_backend(show):
    """Switch to a non-interactive backend when plots are not shown.

    This avoids X/GUI backend issues in headless or multi-process runs.
    """
    if show:
        return
    try:
        plt.switch_backend("Agg")
    except Exception as exc:
        logger.debug(f"Failed to switch matplotlib backend to Agg: {exc}")


class Catchment:
    """Represents a catchment.

    Attributes
    ----------
        name (str): The name of the catchment.
        area (float): The area of the catchment in square units.
    """

    name = ""
    area = ""


class Objectives:
    """A class representing the objectives for evaluating river discharge data.

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
    """Represent a hydrograph and provide methods for calculating metrics and generating plots.

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
        plot_on_axis(function, xvalues, yvalues, colors=None, labels=None, **arguments): Plots multiple graphs.
        check_which_plots_to_create(self, code): Determines which plots to create based on the given code.
        raise_if_not_directory(path): Raises an exception if the given path is not a directory.
        load_data_from_discharge_nc(self, path): Loads discharge data from the specified path.
        load_precipiation_data(self, path): Loads precipitation data from the specified path.
        gen_hydrograph(self, input_path, output_file, show, save, title, plot_code, prec_path): Generates a hydrograph.
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
    # Colorblind-safe defaults: observed is red, simulations avoid reds.
    OBS_COLOR = "#D55E00"
    SIM_COLOR = "#0072B2"
    sim_name = None
    sim_names = None
    multi_sim_named = False
    sim_colors = None

    def __init__(self, simulation=None, observation=None, calc_stats=True):
        self.plots = [0, 0, 0, 0, 0]
        self.calc_stats = calc_stats
        self.sim_discharge_data_list = None
        self.sim_discharge_data_median = None
        self.sim_name = None
        self.sim_names = None
        self.multi_sim_named = False
        self.sim_colors = None
        if simulation is not None and observation is not None:
            self.set_discharge(simulation=simulation, observation=observation)

    def _get_sim_colors(self, n):
        """Return a list of distinct colors for simulations."""
        if self.sim_colors is not None and len(self.sim_colors) >= n:
            return self.sim_colors[:n]
        # Colorblind-safe palette (no reds) to keep observed line distinct.
        colors = [
            self.SIM_COLOR,  # blue
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#CC79A7",  # purple
            "#332288",  # indigo
            "#117733",  # dark green
            "#999933",  # olive
            "#88CCEE",  # light cyan
            "#000000",  # black
        ]
        out = [colors[i % len(colors)] for i in range(n)]
        self.sim_colors = out
        return out

    def _infer_catchment_name(self):
        """Infer catchment name from available data if not set."""
        if self.catchment.name:
            return

        def _from_name(name):
            if not name:
                return None
            ids = re.findall(r"(\d+)", str(name))
            if not ids:
                return None
            return str(int(ids[-1].lstrip("0") or "0"))

        def _from_da(da):
            if da is None:
                return None
            name = _from_name(getattr(da, "name", None))
            if name:
                return name
            if "id" in da.coords:
                try:
                    ids = da.coords["id"].values
                    if np.size(ids) == 1:
                        return str(int(np.asarray(ids).item()))
                except Exception:
                    return None
            return None

        for da in [self.obs_discharge_data, self.sim_discharge_data]:
            name = _from_da(da)
            if name:
                self.catchment.name = name
                return
        if self.sim_discharge_data_list:
            for da in self.sim_discharge_data_list:
                name = _from_da(da)
                if name:
                    self.catchment.name = name
                    return

    def set_discharge(self, simulation=None, observation=None):
        """Set the discharge variables and remove nan values."""
        if simulation is not None and observation is not None:
            self.sim_discharge_data = simulation
            self.obs_discharge_data = observation
        elif simulation is not None or observation is not None:
            msg = "Either one or none of the input must be via array."
            raise ValueError(msg)
        logger.debug(f"Simulation input data: {simulation}")
        logger.debug(f"Observation input data: {observation}")
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
        """Remove empty values from two arrays and return the cleaned data.

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
        """Calculate the Kling-Gupta efficiency and store its components in objectives.

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
        """Calculate the Nash-Sutcliffe efficiency and store it in objectives.

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
        """Clean the input arrays and calculate all objective metrics.

        Args:
            observed : list
                The observed discharge data.
            simulated : list
                The simulated discharge data.

        Returns
        -------
            None
        """
        if self.has_multi_sim():
            if self.sim_discharge_data_median is None:
                self.sim_discharge_data_median = np.nanmedian(
                    np.array(self.sim_discharge_data_list), axis=0
                )
            simulated = self.sim_discharge_data_median
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
        """Find the first unused gridcell for the next plot.

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

    def has_multi_sim(self):
        """Return True if multiple simulation series are configured."""
        return (
            self.sim_discharge_data_list is not None
            and len(self.sim_discharge_data_list) > 1
        )

    def is_last_plot(self, n):
        """Check if the given plot index is the last plot.

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
        """Retrieve the catchment area from the given mHM ConfigFile.log.

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
        """Calculate long-term monthly means for a time-indexed variable.

        Parameters
        ----------
        variable : xarray.DataArray
            Data with a 'time' coordinate.
        long : bool, default False
            If True, prepend last month and append first month (for cyclic plots).

        Returns
        -------
        numpy.ndarray
            Twelve monthly means (or 14 if `long` is True).
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
        """Plot multiple graphs for specified function.

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
            colors = [Hydrograph.SIM_COLOR, Hydrograph.OBS_COLOR]
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
        """Determine which plots to create based on the given code.

        A tuple is produced and saved as member variable. It has a one at the index of the selected plots and a zero otherwise.
        The indices of the plots are:
        model timestep (0), yearly (1), seasonality (2), flow duration (3), scatter (4)

        Args:
            code (str): A string indicating which plots to create.

        Returns
        -------
            None
        """
        if not code:
            self.logger.warning("No plots will be produced since none were specified.")
            return
        if not self.save and not self.show:
            self.logger.info(
                'No plots will be produced since both "save" and "show" are False.'
            )
            return
        if "t" in code:
            self.plots[0] = 1
        if "y" in code:
            self.plots[1] = 1
        if "s" in code:
            self.plots[2] = 1
        if "p" in code:
            self.plots[3] = 1
        if "c" in code and self.calc_stats:
            self.plots[4] = 1

    @staticmethod
    def raise_if_not_directory(path):
        """Raise a NotADirectoryError if the given path is not a directory.

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
        """Load discharge data from the specified path.

        Args:
            path (str): The path to the directory containing the discharge data.

        Raises
        ------
            TypeError: If the variable name in the discharge dataset is not a string.
        """
        path = Path(path)
        discharge_file = path / "discharge.nc" if path.is_dir() else path
        if discharge_file.is_file():
            with get_xarray_ds_from_file(discharge_file) as ds:
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
        """Load precipitation data from a given file or directory path.

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
        with get_xarray_ds_from_file(path) as ds:
            self.pre = ds.load()

    def create_plot_at_timestep(self, fig, gs):
        """Create a discharge plot at the native temporal resolution (daily or hourly).

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
        min_sim_time = None
        max_sim_time = None
        if self.has_multi_sim():
            if self.multi_sim_named and self.sim_names:
                colors = self._get_sim_colors(len(self.sim_discharge_data_list))
                for sim, name, color in zip(
                    self.sim_discharge_data_list, self.sim_names, colors
                ):
                    max_sim_time = (
                        sim.time.max().data
                        if max_sim_time is None or sim.time.max().data < max_sim_time
                        else max_sim_time  # if multiple named simulations, use the earliest max time to set xlim
                    )
                    min_sim_time = (
                        sim.time.min().data
                        if min_sim_time is None or sim.time.min().data < min_sim_time
                        else min_sim_time
                    )
                    ax1.plot(
                        sim.time,
                        sim,
                        color=color,
                        linewidth=0.4,
                        alpha=0.9,
                        label=name,
                    )
            else:
                for sim in self.sim_discharge_data_list:
                    max_sim_time = (
                        sim.time.max().data
                        if max_sim_time is None or sim.time.max().data < max_sim_time
                        else max_sim_time  # if multiple unnamed simulations, use the earliest max time to set xlim
                    )
                    min_sim_time = (
                        sim.time.min().data
                        if min_sim_time is None or sim.time.min().data < min_sim_time
                        else min_sim_time
                    )
                    ax1.plot(
                        sim.time,
                        sim,
                        color="0.7",
                        linewidth=0.4,
                        alpha=0.7,
                        label=None,
                    )
                if self.sim_discharge_data_median is not None:
                    ax1.plot(
                        self.sim_discharge_data_median.time,
                        self.sim_discharge_data_median,
                        color=self.SIM_COLOR,
                        linewidth=0.4,
                        label="sim median",
                    )
            ax1.plot(
                self.obs_discharge_data.time,
                self.obs_discharge_data,
                color=self.OBS_COLOR,
                linewidth=0.4,
                label="observed discharge",
            )
        else:
            max_sim_time = self.sim_discharge_data.time.max().data
            min_sim_time = self.sim_discharge_data.time.min().data
            self.plot_on_axis(
                function=ax1.plot,
                yvalues=[self.sim_discharge_data, self.obs_discharge_data],
                linewidth=0.4,
                labels=[
                    self.sim_name or "simulated discharge",
                    "observed discharge",
                ],
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
            if not self.plots[4]:
                title += f", r = {self.objectives.gamma:.2f}"
        ax1.set_title(
            title,
            horizontalalignment="center",
        )
        ax1.set_ylabel(r"Q $[m^3 s^{-1}]$")
        # set xlim to the common time range of sim and obs, but only within the actual time range of the data (in case one has a shorter time range than the other)
        xmin = max(
            min_sim_time,
            self.obs_discharge_data_nonan.time.min(),
        )
        xmax = min(
            max_sim_time,
            self.obs_discharge_data_nonan.time.max(),
        )
        ax1.set_xlim(xmin, xmax)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
        )

    def create_plot_yearly(self, fig, gs, pre):
        """Generate a yearly discharge plot.

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
        if self.has_multi_sim():
            sim_aligned = xr.align(*self.sim_discharge_data_list, join="inner")
            if self.multi_sim_named and self.sim_names:
                sim_discharge_yearly_list = [
                    sim.resample(time="YE").mean(skipna=True) for sim in sim_aligned
                ]
                sim_discharge_yearly = sim_discharge_yearly_list[0]
                sim_discharge_yearly_median = None
            else:
                sim_stack = xr.concat(sim_aligned, dim="member")
                sim_discharge_yearly = sim_stack.resample(time="YE").mean(skipna=True)
                sim_discharge_yearly_median = sim_discharge_yearly.median(dim="member")
        else:
            sim_discharge_yearly = self.sim_discharge_data_nonan.resample(
                time="YE"
            ).mean(skipna=True)
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
                    f"sum(sim - obs) = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                    horizontalalignment="center",
                )
        else:
            ax2 = fig.add_subplot(outer_gs)
            if self.calc_stats:
                ax2.set_title(
                    f"sum(sim - obs) = {self.objectives.diff:.0f}$m^3$ or {self.objectives.rel_diff*100:.0f}%",
                    horizontalalignment="center",
                )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        if self.has_multi_sim():
            if self.multi_sim_named and self.sim_names:
                colors = self._get_sim_colors(len(sim_discharge_yearly_list))
                for member, name, color in zip(
                    sim_discharge_yearly_list, self.sim_names, colors
                ):
                    ax2.plot(
                        time_yearly_sim,
                        member,
                        color=color,
                        linewidth=0.5,
                        alpha=0.9,
                        label=name,
                    )
            else:
                for member in sim_discharge_yearly:
                    ax2.plot(
                        time_yearly_sim,
                        member,
                        color="0.7",
                        linewidth=0.6,
                        alpha=0.7,
                    )
                if sim_discharge_yearly_median is not None:
                    ax2.plot(
                        [int(y.dt.year.data) for y in sim_discharge_yearly_median.time],
                        sim_discharge_yearly_median,
                        color=self.SIM_COLOR,
                        linewidth=0.6,
                        label="sim median",
                    )
            ax2.plot(
                time_yearly_obs,
                obs_discharge_yearly,
                color=self.OBS_COLOR,
                linewidth=0.5,
                label="observed discharge",
            )
        else:
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
                linewidth=0.6,
                labels=[
                    self.sim_name or "simulated discharge",
                    "observed discharge",
                ],
            )
        if r == 0:
            ax2.legend()
        ax2.set_ylabel(r"Q $[m^3 s^{-1}]$")
        ax2.set_xlim(np.min(years_combined), np.max(years_combined))
        ax2.set_xticks(
            years_combined[:: len((years_combined) - np.min(years_combined)) // 3]
        )

    def create_plot_seasonality(self, fig, gs, pre):
        """Generate a discharge seasonality plot.

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The grid specification for the plot layout.
            pre (dict): The precipitation data.

        Returns
        -------
            None
        """
        self.logger.info("generating discharge seasonality plot")

        if self.has_multi_sim():
            sim_aligned = xr.align(*self.sim_discharge_data_list, join="inner")
            season_sim_all = [
                self.get_long_time_monthly_mean(sim_aligned[i], long=True)
                for i in range(len(sim_aligned))
            ]
            season_sim_all = np.array(season_sim_all)
            if self.multi_sim_named and self.sim_names:
                season_sim = season_sim_all[0]
            else:
                season_sim = np.nanmedian(season_sim_all, axis=0)
        else:
            season_sim = self.get_long_time_monthly_mean(
                self.sim_discharge_data, long=True
            )
        # check that there are at least 4 monthly values that are not nan in both sim and obs to create the plot
        if (
            np.sum(~np.isnan(season_sim)) < 4
            or np.sum(~np.isnan(self.obs_discharge_data)) < 4
        ):
            logger.warning(
                "Cannot create seasonality plot because there are less than 4 monthly values that are not nan in either sim or obs."
            )
            return
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
        season_obs = self.get_long_time_monthly_mean(self.obs_discharge_data, long=True)
        self.logger.debug(f"sim: {season_sim}")
        self.logger.debug(f"osb: {season_obs}")
        if self.has_multi_sim():
            if self.multi_sim_named and self.sim_names:
                colors = self._get_sim_colors(len(season_sim_all))
                for series, name, color in zip(season_sim_all, self.sim_names, colors):
                    ax3.plot(
                        np.arange(0, 14),
                        series,
                        color=color,
                        linewidth=0.5,
                        alpha=0.9,
                        label=name,
                    )
            else:
                for series in season_sim_all:
                    ax3.plot(
                        np.arange(0, 14),
                        series,
                        color="0.7",
                        linewidth=0.6,
                        alpha=0.7,
                    )
                ax3.plot(
                    np.arange(0, 14),
                    season_sim,
                    color=self.SIM_COLOR,
                    linewidth=0.6,
                    label="sim median",
                )
            ax3.plot(
                np.arange(0, 14),
                season_obs,
                color=self.OBS_COLOR,
                linewidth=0.5,
                label="observed discharge",
            )
        else:
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
                linewidth=0.6,
                labels=[
                    self.sim_name or "simulated discharge",
                    "observed discharge",
                ],
            )

        if r == 0:
            ax3.legend()

        ax3.set_xlim(0.5, 12.5)
        ax3.set_xticks(
            np.arange(1, 13),
        )
        ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")

    def create_plot_flow_duration(self, fig, gs):
        """Create a flow duration plot.

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The gridspec object specifying the subplot layout.

        Returns
        -------
            None
        """
        self.logger.info("generating flow duration plot")
        r, c = self.get_row_col()
        if r == 0 or self.is_last_plot(3):
            ax_p = fig.add_subplot(gs[r, c:])
            self.grid[r] = [True for _ in self.grid[r]]
        else:
            ax_p = fig.add_subplot(gs[r, c])
            self.grid[r][c] = True

        if self.has_multi_sim():
            sim_series = [
                np.array(sim).flatten() for sim in self.sim_discharge_data_list
            ]
            sim_series = [s[~np.isnan(s)] for s in sim_series]
            if self.multi_sim_named and self.sim_names:
                sim_median = None
            else:
                sim_median = np.array(self.sim_discharge_data_median).flatten()
                sim_median = sim_median[~np.isnan(sim_median)]
        else:
            sim_median = np.array(self.sim_discharge_data_nonan).flatten()
            sim_median = sim_median[~np.isnan(sim_median)]
        obs = np.array(self.obs_discharge_data_nonan).flatten()
        obs = obs[~np.isnan(obs)]
        if not (self.multi_sim_named and self.sim_names) and (
            sim_median.size == 0 or obs.size == 0
        ):
            self.logger.warning("Flow duration plot skipped: no valid discharge data.")
            ax_p.set_axis_off()
            return
        if self.has_multi_sim():
            sim_sorted_all = [np.sort(s)[::-1] for s in sim_series if s.size > 0]
            if self.multi_sim_named and self.sim_names:
                colors = self._get_sim_colors(len(sim_sorted_all))
                for series, name, color in zip(sim_sorted_all, self.sim_names, colors):
                    sim_p = (np.arange(1, series.size + 1) / (series.size + 1)) * 100.0
                    ax_p.plot(
                        sim_p,
                        series,
                        color=color,
                        linewidth=0.6,
                        alpha=0.9,
                        label=name,
                    )
            else:
                for series in sim_sorted_all:
                    sim_p = (np.arange(1, series.size + 1) / (series.size + 1)) * 100.0
                    ax_p.plot(sim_p, series, color="0.7", linewidth=0.6, alpha=0.7)
                sim_sorted = np.sort(sim_median)[::-1]
                sim_p = (
                    np.arange(1, sim_sorted.size + 1) / (sim_sorted.size + 1)
                ) * 100.0
                ax_p.plot(
                    sim_p,
                    sim_sorted,
                    color=self.SIM_COLOR,
                    linewidth=0.6,
                    label="sim median",
                )
            obs_sorted = np.sort(obs)[::-1]
            obs_p = (np.arange(1, obs_sorted.size + 1) / (obs_sorted.size + 1)) * 100.0
            ax_p.plot(
                obs_p,
                obs_sorted,
                color=self.OBS_COLOR,
                linewidth=0.5,
                label="observed discharge",
            )
        else:
            sim_sorted = np.sort(sim_median)[::-1]
            obs_sorted = np.sort(obs)[::-1]
            sim_p = (np.arange(1, sim_sorted.size + 1) / (sim_sorted.size + 1)) * 100.0
            obs_p = (np.arange(1, obs_sorted.size + 1) / (obs_sorted.size + 1)) * 100.0
            self.plot_on_axis(
                function=ax_p.plot,
                xvalues=[sim_p, obs_p],
                yvalues=[sim_sorted, obs_sorted],
                linewidth=0.6,
                labels=[
                    self.sim_name or "simulated discharge",
                    "observed discharge",
                ],
            )
        if r == 0:
            ax_p.legend()
        ax_p.set_xlim(-0.5, 100)
        ax_p.set_xticks([0, 25, 50, 75, 100])
        ax_p.set_ylim(-0.5, None)
        ax_p.set_xlabel("% time flow equaled or exceeded")
        ax_p.set_ylabel(r"Q $[m^3 s^{-1}]$")
        ax_p.set_title("Flow duration", horizontalalignment="center")
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)

    def create_plot_scatter(self, fig, gs):
        """Create a discharge plot at the native temporal resolution (daily or hourly).

        Args:
            fig (matplotlib.figure.Figure): The figure object to add the plot to.
            gs (matplotlib.gridspec.GridSpec): The gridspec object specifying the subplot layout.

        Returns
        -------
            None
        """
        self.logger.info("generating discharge scatter plot")
        r, c = self.get_row_col()
        if r == 0 or self.is_last_plot(4):
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
            linewidth=0.6,
            alpha=0.1,
            labels=[self.sim_name or "simulated discharge"],
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
                f"In crop data: obs is nan {t1.any()} or sim is nan {t2.any()}"
            )
            raise ValueError(only_nan_msg)
        logger.info(f"sim_discharge_data: {self.sim_discharge_data.data}")
        logger.info(f"obs_discharge_data: {self.obs_discharge_data.data}")
        if self.pre is not None:
            self.pre = self.pre.sel(time=slice(start, end))
        return True

    def get_hydrograph(self):
        """Generate the hydrograph from the data stored in member variables.

        Returns
        -------
        bool | None
            True on success, False/None if no plots are created or data are insufficient.
        """
        _ensure_non_interactive_backend(self.show)
        # calculate metrics at timestep resolution (generally hourly)
        if self.calc_stats:
            logger.info("Crop to overlapping time.")
            if not self.crop_data_to_overlapping_time():
                return False
            logger.info("Calculate objectives")
            if not self.calc_objectives(
                self.obs_discharge_data, self.sim_discharge_data
            ):
                return False
        if sum(self.plots) == 0:
            self.logger.warning("Create no plots")
            return True
        self._infer_catchment_name()
        # create figure and determining the number of rows and cols
        fig = plt.figure(figsize=(7, 8))
        nrows = sum(self.plots) // 2 + 1
        ncols = 2
        self.logger.debug(f"nrows = {nrows} and ncols = {ncols}")

        # generate a grid indicating used and unused cells
        self.grid = [[False] * ncols for _ in range(nrows)]
        gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1])

        # write title
        logger.info(f"catchment name: {self.catchment.name}")
        fig.text(
            s=f"{self.catchment.name}",
            x=0.01,
            y=0.97,
            horizontalalignment="left",
            fontsize="x-large",
        )
        if self.catchment.area:
            fig.text(
                s=f"Area = {self.catchment.area}" + r"$km^2$",
                x=0.5,
                y=0.97,
                horizontalalignment="center",
                fontsize="x-large",
            )
        suptitle_text = f"\n{self.title}\n"
        # If seasonality is the only selected plot, show only alpha and beta in suptitle.
        if self.calc_stats and self.plots[2] and sum(self.plots) == 1:
            stats = (
                f"alpha = {self.objectives.alpha:.2f}, "
                f"beta = {self.objectives.beta:.2f}"
            )
            suptitle_text = (
                f"\n{self.title}\n{stats}\n" if self.title else f"\n{stats}\n"
            )
        fig.suptitle(
            t=suptitle_text,
            x=0.5,
            y=0.97,
            horizontalalignment="center",
            fontsize="x-large",
        )

        # generate plots
        if self.plots[0]:
            self.create_plot_at_timestep(fig, gs)
        if self.plots[1]:
            self.create_plot_yearly(fig, gs, self.pre)
        if self.plots[2]:
            self.create_plot_seasonality(fig, gs, self.pre)
        if self.plots[3]:
            self.create_plot_flow_duration(fig, gs)
        if self.plots[4]:
            self.create_plot_scatter(fig, gs)
        plt.tight_layout()
        if self.save:
            # if (
            #     len(self.output_file.split("/")) == 1
            # ):  # by default the hydrograph is saved to the data directory
            # self.output_file = self.output_file
            if not self.output_file.parent.is_dir():
                self.output_file.parent.mkdir(parents=True)
            fig.savefig(self.output_file, bbox_inches="tight", dpi=800)
            self.logger.info(f"saved hydrograph to {self.output_file}")
        if self.show:
            plt.show()
        else:
            plt.close()
        return True

    def write_output(self):
        """Write calculated objective metrics to CSV."""
        out_dict = {k: [v] for k, v in {**self.objectives.__dict__}.items()}
        if self.catchment.name is not None:
            out_dict["id"] = str(self.catchment.name)
        logger.info(f"generated metrics: {dict_to_multiline_string(out_dict)}")
        create_csv_from_dict(out_dict, self.output_file.parent / "kge.csv")


@log_arguments()
def get_hydrograph_from_path(  # noqa: PLR0912, PLR0915
    input_path, output_file, show, save, title, plot_code, prec_path, sim_names=None
):
    """Read discharge data and produce a hydrograph with multiple analyses.

    Simulated and observed discharge are plotted for different temporal
    resolutions. Additionally, a seasonality plot and a scatter plot
    (simulated versus observed) are generated.
    """
    input_paths = (
        list(input_path) if isinstance(input_path, (list, tuple)) else [input_path]
    )
    multi_input = len(input_paths) > 1
    input_path = input_paths if multi_input else input_paths[0]

    def _normalize_names(names):
        if names is None:
            return None
        if isinstance(names, str):
            names = [names]
        names = list(names)
        if len(names) == 1 and "," in names[0]:
            names = [n.strip() for n in names[0].split(",") if n.strip()]
        return names or None

    sim_names = _normalize_names(sim_names)
    named_multi = bool(multi_input and sim_names and len(sim_names) == len(input_paths))
    if multi_input and sim_names and not named_multi:
        logger.warning(
            f"Provided {len(sim_names)} names but {len(input_paths)} input paths; "
            "ignoring names."
        )
        sim_names = None

    hydro = Hydrograph(calc_stats=not named_multi)
    hydro.show = show
    hydro.save = save
    hydro.check_which_plots_to_create(plot_code)
    if multi_input:
        hydro.plots[4] = 0
    output_file = Path(output_file)
    if output_file.is_dir():
        hydro.output_file = output_file / "hydrograph.png"
    elif output_file.suffix:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        hydro.output_file = output_file
    else:  # is non existing directory
        output_file.mkdir(parents=True)
        hydro.output_file = output_file / "hydrograph.png"

    hydro.title = title
    if not multi_input and sim_names:
        hydro.sim_name = sim_names[0]
    prec_path = Path(prec_path)
    if multi_input:
        input_paths = [Path(p) for p in input_path]
        sims = []
        obs = None
        obs_var_name = None
        for path in input_paths:
            discharge_file = path / "discharge.nc" if path.is_dir() else path
            if not discharge_file.is_file():
                msg = f"No discharge.nc file at {path}"
                with ErrorLogger():
                    raise ValueError(msg)
            with get_xarray_ds_from_file(discharge_file) as ds:
                discharge_data = ds.load()
                sim_var = None
                obs_var = None
                for v in discharge_data.variables:
                    if not isinstance(v, str):
                        msg = f"variable name is not a string - {v} - {type(v)}"
                        raise TypeError(msg)
                    if "sim" in v:
                        sim_var = v
                    if "obs" in v:
                        obs_var = v
                if obs_var is None or sim_var is None:
                    msg = f"Missing sim/obs variables in {discharge_file}"
                    with ErrorLogger():
                        raise ValueError(msg)
                if obs_var_name is None:
                    obs_var_name = obs_var
                    obs = discharge_data[obs_var]
                    # derive catchment name from obs variable (e.g. Qobs_0006342210)
                    if hydro.catchment.name is None:
                        ids = re.findall(r"(\d+)", obs_var_name)
                        if ids:
                            hydro.catchment.name = str(int(ids[-1].lstrip("0") or "0"))
                elif obs_var_name != obs_var:
                    msg = (
                        "Observed discharge variable name differs between inputs: "
                        f"{obs_var_name} vs {obs_var}"
                    )
                    with ErrorLogger():
                        raise ValueError(msg)
                sims.append(discharge_data[sim_var])
        sims_aligned = xr.align(*sims, join="inner")
        if named_multi:
            hydro.multi_sim_named = True
            hydro.sim_names = sim_names
            hydro.sim_discharge_data_list = list(sims_aligned)
            hydro.sim_discharge_data_median = None
            hydro.set_discharge(simulation=sims_aligned[0], observation=obs)
        else:
            sim_stack = xr.concat(sims_aligned, dim="member")
            sim_median = sim_stack.median(dim="member")
            hydro.set_discharge(simulation=sim_median, observation=obs)
            hydro.sim_discharge_data_list = list(sims_aligned)
            hydro.sim_discharge_data_median = sim_median
        hydro.get_catchment_area(input_paths[0], ndecimal=0)
    else:
        input_path = Path(input_path)
        if not hydro.load_data_from_discharge_nc(input_path):
            msg = f"No discharge.nc file at {input_path}"
            with ErrorLogger():
                raise ValueError(msg)
        hydro.get_catchment_area(input_path, ndecimal=0)
    hydro.load_precipiation_data(prec_path)
    hydro.get_hydrograph()
    if named_multi:
        # compute per-simulation metrics and write to csv
        catchment_id = None
        if obs_var_name:
            ids = re.findall(r"(\d+)", obs_var_name)
            if ids:
                catchment_id = int(ids[-1].lstrip("0") or "0")
        metrics_rows = []
        for sim, name in zip(hydro.sim_discharge_data_list, sim_names):
            h = Hydrograph(calc_stats=True)
            if not h.set_discharge(simulation=sim, observation=obs):
                continue
            try:
                if not h.crop_data_to_overlapping_time():
                    continue
                if not h.calc_objectives(h.obs_discharge_data, h.sim_discharge_data):
                    continue
            except Exception:
                continue
            row = {**h.objectives.__dict__}
            if catchment_id is not None:
                row["id"] = catchment_id
            row["name"] = name
            metrics_rows.append(row)
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(
                hydro.output_file.parent / "kge.csv", index=False
            )
    else:
        hydro.write_output()


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
    raise_exceptions=True,
    **kwargs,
):
    """Create a hydrograph from xarray datasets for discharge and precipitation.

    Simulated and observed discharge are plotted at multiple temporal
    resolutions. Additionally, a seasonality plot and a scatter plot
    (simulated versus observed) are generated.
    """
    hydro = Hydrograph(calc_stats=calc_stats)
    try:
        missing_data_error_msg = f"For {id} the hydrograph could not be created."
        if hydro.set_discharge(simulation=simulations, observation=observation):
            hydro.pre = precipitation
            hydro.output_file = Path(output_file)
            hydro.title = str(id) if not title and id is not None else title
            hydro.show = show
            hydro.save = save
            hydro.catchment.area = area
            hydro.check_which_plots_to_create(plot_code)
            if not hydro.get_hydrograph():
                logger.debug(
                    f"get_hydrograph returned False for {id}. Simulations: {simulations}, Observation: {observation}"
                )
                logger.error(missing_data_error_msg)
        else:
            logger.debug(
                f"set_discharge returned False for {id}. Simulations: {simulations}, Observation: {observation}"
            )
            logger.error(missing_data_error_msg)
    except Exception as e:
        logger.error(e)
        if raise_exceptions:
            raise e
    return {**hydro.objectives.__dict__, "id": id, **kwargs}
