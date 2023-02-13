"""
26.01.2023
"""

import getopt
import itertools
import sys

import hydroeval as he
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr


def get_catchment_area(path, ndecimal=0):
    try:
        with open(path + "ConfigFile.log", "r") as f:
            doc = f.readlines()
            for l in doc[::-1]:
                if l.strip():
                    return f"{float(l.replace('Total[km2]', '').strip()):.{ndecimal}f}"
    except:
        print("Area could not be read.")
        return ""


def get_long_time_monthly_mean(variable):
    """ Takes a variable and calculated the long time average value for every month of the year
    :param variable: xarray with a variable and a time
    :return: list of twelve numbers corresponding to the long term average value for each month of the year
    """
    var_ses = [[], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(len(variable)):
        var_ses[int(variable.time[i].dt.month.data) - 1].append(variable[i])
    return np.array([np.mean(np.array(m)) for m in var_ses])


def plot_on_axis(
    function, xvalues, yvalues: list, colors=None, labels=None, **arguments
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

def check_which_plots_to_create(a):
    a = int(a)
    lst = list(itertools.product([0, 1], repeat=4))
    for l in lst:
        check = 0
        for i, v in enumerate(l):
            check += v * 2**i
        if a == check:
            return l
    return (0,0,0,0)
def gen_hydrograph(input_path, output_path, filename, show, save, title, plots):
    """
    Read in discharge data and plot the simulated against the observed discharge for different time resolutions and a seasonality.
    :param input_path: Path to discharge.nc file
    :param output_path: Path where the hydrograph png will be saved
    :param filename: Filename of the resulting file. e.g. hydrograph.png
    :param show: bool if plots should be shown or not
    :param save: bool if plots should be saved or not
    :param title: title given to the hydrograph
    """
    if output_path is None:
        output_path = input_path
    plots = check_which_plots_to_create(plots)
    if sum(plots) == 0:
        return
    with xr.open_dataset(input_path + "discharge.nc") as ds:
        discharge_daily = ds.load()
        for v in discharge_daily.variables:
            for key in ["sim", "obs"]:
                if key in v:
                    catchment = str(int(v.split("_")[1]))
                    discharge_daily = discharge_daily.rename({v: key})
        discharge_monthly = discharge_daily.resample(time="M").mean(skipna=False)
        discharge_yearly = discharge_daily.resample(time="Y").mean(skipna=False)

        nse_daily = he.evaluator(he.nse, discharge_daily["sim"], discharge_daily["obs"])
        kge_daily, r_daily, alpha_daily, beta_daily = he.evaluator(
            he.kge, discharge_daily["sim"], discharge_daily["obs"]
        )
        nse_monthly = he.evaluator(
            he.nse, discharge_monthly["sim"], discharge_monthly["obs"]
        )
        kge_monthly, r_monthly, alpha_monthly, beta_monthly = he.evaluator(
            he.kge, discharge_monthly["sim"], discharge_monthly["obs"]
        )
        nse_yearly = he.evaluator(
            he.nse, discharge_yearly["sim"], discharge_yearly["obs"]
        )
        kge_yearly, r_yearly, alpha_yearly, beta_yearly = he.evaluator(
            he.kge, discharge_yearly["sim"], discharge_yearly["obs"]
        )

        fig = plt.figure()
        fig.set_size_inches(7, 8)
        nrows = sum(plots)//2+1
        ncols = ncols = 2 if sum(plots) > 2 else 1
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
        area = get_catchment_area(input_path, ndecimal=0)
        fig.text(
            s=f"{catchment}",
            x=0.01,
            y=0.97,
            horizontalalignment="left",
            fontsize="x-large",
        )
        fig.text(
            s=f"Area = " + area + r"$km^2$",
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
        row = 0
        column = 0
        if plots[0]:
            ax1 = fig.add_subplot(gs[row, :])
            row += 1
            plot_on_axis(
                function=ax1.plot,
                xvalues=discharge_daily["time"],
                yvalues=[discharge_daily["sim"], discharge_daily["obs"]],
                linewidth=1.0,
            )
            ax1.legend()
            ax1.set_title(
                f"Daily:  KGE = {kge_daily[0]:.2f} , NSE = {nse_daily[0]:.2f}",
                horizontalalignment="center",
            )  # , alpha_day={alpha_daily[0]:.2f} , beta_day={beta_daily[0]:.2f} , r_day={r_daily[0]:.2f}')
            ax1.set_ylabel(r"Q $[m^3 s^{-1}]$")

        if plots[1]:
            if row < nrows - 1:
                ax2 = fig.add_subplot(gs[row, :])
                row += 1
            else:
                ax2 = fig.add_subplot(gs[row, column])
                column += 1
            plot_on_axis(
                function=ax2.plot,
                xvalues=discharge_monthly["time"],
                yvalues=[discharge_monthly["sim"], discharge_monthly["obs"]],
                linewidth=1.0,
            )
            ax2.set_title(
                f"Monthly:  KGE = {kge_monthly[0]:.2f} , NSE = {nse_monthly[0]:.2f}",
                horizontalalignment="center",
            )
            if row == 0:
                ax2.legend()
            # ax2.xaxis.set_major_formatter(
            #     mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
            # ax2.xaxis.set_tick_params() #.locator_params(axis='x', nbins=9//column)
            # ax2.set_xticks(discharge_monthly['time'] // 2)
            ax2.set_ylabel(r"Q $[m^3 s^{-1}]$")

        if plots[2]:
            if row < nrows - 1:
                ax3 = fig.add_subplot(gs[row, :])
                row += 1
            else:
                ax3 = fig.add_subplot(gs[row, column])
                column += 1
            time_yearly = [str(y.dt.year.data) for y in discharge_yearly["time"]]
            plot_on_axis(
                function=ax3.plot,
                xvalues=time_yearly,
                yvalues=[discharge_yearly["sim"], discharge_yearly["obs"]],
                linewidth=1.0,
            )
            if row == 0:
                ax3.legend()
            ax3.set_title(
                f"Yearly:  KGE = {kge_yearly[0]:.2f} , NSE = {nse_yearly[0]:.2f}",
                horizontalalignment="center",
            )
            ax3.set_ylabel(r"Q $[m^3 s^{-1}]$")
            ax3.set_xticks(time_yearly[:: len(time_yearly) // 3])

        if plots[3]:
            if row < nrows - 1:
                ax4 = fig.add_subplot(gs[row, :])
                row += 1
            else:
                ax4 = fig.add_subplot(gs[row, column])
                column += 1
            plot_on_axis(
                function=ax4.scatter,
                xvalues=range(1, 13),
                yvalues=[
                    get_long_time_monthly_mean(discharge_daily["sim"]),
                    get_long_time_monthly_mean(discharge_daily["obs"]),
                ],
                s=1,
            )
            plot_on_axis(
                function=ax4.plot,
                xvalues=range(1, 13),
                yvalues=[
                    get_long_time_monthly_mean(discharge_daily["sim"]),
                    get_long_time_monthly_mean(discharge_daily["obs"]),
                ],
                linewidth=0.5,
            )
            if row == 0:
                ax4.legend()
            ax4.set_title(f"Seasonality", horizontalalignment="center")
            ax4.set_xlim(0.8, 12.2)
            ax4.set_ylabel(r"Q $[m^3 s^{-1}]$")
            ax4.set_xlabel(r"month")

        plt.tight_layout()
        if save:
            fig.savefig(output_path + filename, bbox_inches="tight")
            print(f"saved hydrograph to '{output_path + filename}'")
        if show:
            plt.show()
        else:
            plt.close()