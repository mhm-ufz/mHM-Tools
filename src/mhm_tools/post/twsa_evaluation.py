"""Evaluate a simulated total water storage anomaly against a gridded reference.

The tool reads a model record and a reference record from files or from
directories that are searched recursively, brings them onto a common grid and the
calendar of the reference, reduces both to an anomaly against one baseline period and
scores every grid cell with the selected metrics.

Input formats
-------------
- ``mhm_states``: one or more mHM fluxes-and-states files. The storage
  variables (interception, swe, the SWC_L layers, sealedSTW, unsatSTW and
  satSTW) are converted to mm and summed into the total water storage.
- ``tws``: total water storage, used as it is.
- ``twsa``: an already computed total water storage anomaly. The baseline of
  such a record is unknown, so it is assumed to match the requested one.

Normalization rules
-------------------
- Grid: the finer dataset is aggregated onto the coarser grid by spatial mean,
  so the model is upscaled onto the reference grid in the usual case that the
  reference is coarser. Upsampling never happens, because it would invent
  information.
- Calendar: the reference sets the calendar and keeps every one of its steps,
  and the input is averaged over the window each reference value covers. A
  reference that states those windows through CF time bounds is taken at its
  word, which is what a satellite product needs whose steps drift against the
  calendar or leave gaps. A reference without bounds has its windows derived
  from its stamps, which only holds while they are evenly spaced; a gap makes
  the record state its own bounds instead. Windows that the input does not
  cover completely are dropped.
- Anomaly: the mean of the baseline period is subtracted per grid cell, so the
  seasonal cycle stays in the signal. A record that does not cover the whole
  baseline period is an error rather than a weaker baseline.

Reading the KGE of an anomaly
-----------------------------
The bias ratio of the KGE divides the two record means. Both records are
anomalies whose baseline mean is zero by construction, so dividing them
directly turns the bias into noise that explodes wherever the reference mean
happens to sit close to zero. The storage level the anomalies were taken from
is therefore added back to both records before they are scored: the reference
baseline mean, the input baseline mean when the reference arrived as an
anomaly, or a constant when neither record left a baseline behind. A common
offset leaves the correlation and the variability ratio untouched and turns
beta into ``(input mean + offset) / (reference mean + offset)``, a ratio of two
storages again. The offset is written next to the metrics, so beta stays
readable, and every component is written separately so a bias that is still
unstable stays visible instead of silently sinking the score.

Outputs
-------
- ``twsa_metrics_<input>_vs_<ref>.nc`` with the selected metrics: the KGE
  with ``alpha``, ``beta``, ``gamma``, the compared month count, the reference
  mean and standard deviation and the baseline offset that make the bias term
  readable, and the root mean square error.
- ``twsa_metric_<name>_<input>_vs_<ref>.png`` per selected metric and per KGE
  component.
- ``twsa_monthly_<name>.nc`` per dataset with the normalized anomaly on the
  shared calendar.
- ``twsa_cells_<input>_vs_<ref>_region_<region>.png`` per evaluation region,
  with every grid cell over time and the region mean of both datasets.
- the KGE map again per evaluation region, where the KGE is selected.

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
from matplotlib.collections import LineCollection

from mhm_tools.common.constants import (
    KGE_CONSTANT_MEAN_BOUND,
    MHM_SOIL_MOISTURE_PREFIX,
    MHM_TWS_STORAGE_VARS,
    WMO_REGION_BOUNDS,
)
from mhm_tools.common.file_handler import (
    ChunkType,
    get_dataset_from_path,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.metrics.kge import calculate_kling_gupta_efficiency_per_cell
from mhm_tools.common.metrics.rmse import calculate_root_mean_square_error_per_cell
from mhm_tools.common.plotter import (
    CAPTION_COLOR,
    INPUT_COLOR,
    REF_COLOR,
    plot_map,
    style_axes,
)
from mhm_tools.common.time_utils import (
    get_time_bounds,
    resample_to_reference_windows,
    set_time_bounds,
)
from mhm_tools.common.utils import format_region_title, sanitize_name, select_regions
from mhm_tools.common.xarray_utils import (
    calculate_anomaly,
    convert_water_storage_to_mm,
    crop_to_region,
    get_overlapping_time_slice,
    get_single_data_var,
    materialize_if_within_budget,
    normalize_lat_lon,
    normalize_time,
    regrid_to_coarser_grid,
)
from mhm_tools.post.gridded_data_evaluation import crop_datasets_to_spatial_overlap

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_START_YEAR = 2004
DEFAULT_BASELINE_END_YEAR = 2009
DEFAULT_MIN_VALID_MONTHS = 24

INPUT_FORMATS = ("mhm_states", "tws", "twsa")
AUTO_FORMAT = "auto"
SOIL_LAYER_PATTERN = re.compile(rf"^{MHM_SOIL_MOISTURE_PREFIX}\d+$", re.IGNORECASE)
# an anomaly marker wins, because a GRACE standard name reads
# "lwe_thickness_of_water_storage_anomaly" and holds both markers
ANOMALY_NAME_MARKERS = ("anom",)
STORAGE_NAME_MARKERS = ("tws", "storage", "lwe")

# the discharge evaluation scales a KGE the same way
KGE_VMIN = KGE_CONSTANT_MEAN_BOUND
KGE_VMAX = 1.0
# cells scoring below the constant mean value bound get their own colour
KGE_UNDER_COLOR = "lightgray"
# below this share of its own spread a bias denominator makes the ratio noise
ZERO_MEAN_BETA_TOLERANCE = 0.01
# storage level in mm added back to both anomalies before they are scored, used
# when neither record left a baseline mean behind
DEFAULT_KGE_BASELINE_OFFSET = 1000.0

DEFAULT_MAX_REGION_CELL_LINES = 2000
# metrics a run can score its cells with, selected per run
AVAILABLE_METRICS = ("kge", "rmse")
# components the KGE is written and plotted with
KGE_COMPONENTS = ("alpha", "beta", "gamma")
CELL_LINE_WIDTH = 0.4
REGION_MEAN_LINE_WIDTH = 1.8


# ---------------------------------------------------------------------------
# reading and format detection
# ---------------------------------------------------------------------------


def get_mhm_storage_variables(ds):
    """Collect the mHM storage variables a dataset holds.

    Args:
        ds: Dataset to inspect.

    Returns
    -------
        List of the storage variable names, the soil layers sorted by their
        layer number and appended. Empty when the dataset holds none of them.
    """
    present = [name for name in MHM_TWS_STORAGE_VARS if name in ds.data_vars]
    layers = sorted(
        name for name in ds.data_vars if SOIL_LAYER_PATTERN.match(str(name))
    )
    return present + layers


def detect_input_format(ds, label, var_name=None, forced_format=AUTO_FORMAT):
    """Resolve which of the three input formats a dataset holds.

    A forced format wins. Otherwise the mHM storage variables are looked for
    first, then the variable name and its description are searched for an
    anomaly and a storage marker. An undecidable dataset is an error, because
    guessing it wrong silently evaluates the wrong quantity.

    Args:
        ds: Dataset to inspect.
        label: Name used in the log and error messages.
        var_name: Variable to inspect, or None to detect the single one.
        forced_format: One of `INPUT_FORMATS`, or "auto" to detect it.

    Returns
    -------
        The format name, one of `INPUT_FORMATS`.
    """
    if forced_format and forced_format != AUTO_FORMAT:
        if forced_format not in INPUT_FORMATS:
            msg = (
                f"Unknown {label} format {forced_format!r}. "
                f"Available formats: {', '.join(INPUT_FORMATS)}."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(f"Using the given {label} format '{forced_format}'.")
        return forced_format

    if get_mhm_storage_variables(ds):
        logger.info(f"Detected the {label} format 'mhm_states'.")
        return "mhm_states"

    name = var_name or get_single_data_var(ds)
    if name is None:
        msg = (
            f"Could not determine the {label} variable automatically. "
            f"Available data variables: {list(ds.data_vars)}. Pass it explicitly."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    attrs = ds[name].attrs
    markers = " ".join(
        str(part).lower()
        for part in (
            name,
            attrs.get("long_name", ""),
            attrs.get("standard_name", ""),
            attrs.get("description", ""),
        )
    )
    if any(marker in markers for marker in ANOMALY_NAME_MARKERS):
        logger.info(f"Detected the {label} format 'twsa' from '{name}'.")
        return "twsa"
    if any(marker in markers for marker in STORAGE_NAME_MARKERS):
        logger.info(f"Detected the {label} format 'tws' from '{name}'.")
        return "tws"
    msg = (
        f"Could not tell which format the {label} variable '{name}' holds. Its "
        f"name and its long_name, standard_name and description attributes hold "
        f"neither a storage nor an anomaly marker. Pass the format explicitly "
        f"as one of {', '.join(INPUT_FORMATS)}."
    )
    with ErrorLogger(logger):
        raise ValueError(msg)


def calculate_total_water_storage(ds, storage_variables, scale_factor, label):
    """Convert every mHM storage variable to mm and sum them into one field.

    Every listed storage is required, because summing a subset yields a
    plausible looking but wrong total. The sum propagates missing values, so a
    cell missing one component has no total instead of an understated one.

    Args:
        ds: Dataset holding the mHM storage variables.
        storage_variables: Names to sum, as returned by
            `get_mhm_storage_variables`.
        scale_factor: Millimetres per data unit, or None to use the units.
        label: Name used in the log and error messages.

    Returns
    -------
        The total water storage DataArray in mm, named "tws".
    """
    missing = [name for name in MHM_TWS_STORAGE_VARS if name not in ds.data_vars]
    layers = [name for name in storage_variables if SOIL_LAYER_PATTERN.match(str(name))]
    if missing or not layers:
        msg = (
            f"The {label} fluxes-and-states dataset cannot be summed into a total "
            f"water storage. Missing storages: {missing or 'none'}. Soil layers "
            f"found: {layers or 'none'}. Expected every one of "
            f"{', '.join(MHM_TWS_STORAGE_VARS)} and at least one "
            f"{MHM_SOIL_MOISTURE_PREFIX}xx layer."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Summing the {label} storages {', '.join(storage_variables)} into the "
        f"total water storage."
    )
    total = None
    for name in storage_variables:
        converted = convert_water_storage_to_mm(
            ds[name], scale_factor=scale_factor, label=f"{label} '{name}'"
        )
        total = converted if total is None else total + converted
    total.name = "tws"
    total.attrs = {
        "units": "mm",
        "long_name": "total water storage",
        "description": f"sum of {', '.join(storage_variables)}",
    }
    return total


def read_twsa_data_array(
    data_path,
    file_name,
    var_name,
    forced_format,
    scale_factor,
    label,
    max_memory_gib=8.0,
):
    """Read a storage record and return it in mm together with its format.

    The record is chunked along time rather than read as one block, so a long
    record streams through the evaluation instead of having to fit in memory.

    Args:
        data_path: Path to a NetCDF file or a directory searched recursively.
        file_name: Glob pattern used for the recursive directory search.
        var_name: Variable to select, or None to detect it.
        forced_format: One of `INPUT_FORMATS`, or "auto" to detect it.
        scale_factor: Millimetres per data unit, or None to use the units.
        label: Name used in the log and error messages.
        max_memory_gib: Memory budget one chunk is sized against.

    Returns
    -------
        Tuple of the DataArray in mm with ``time``, ``lat`` and ``lon``, and the
        resolved format name.
    """
    logger.info(f"Reading {label} data from {data_path}")
    # a fluxes-and-states file holds dozens of variables, so the storages are
    # named on a second read once the format is known
    probe = get_dataset_from_path(
        data_path,
        file_name=file_name,
        normalize_latlon_coords=True,
        use_mfdataset=True,
        force_ascending_y=True,
    )
    input_format = detect_input_format(probe, label, var_name, forced_format)
    storage_variables = (
        get_mhm_storage_variables(probe) if input_format == "mhm_states" else None
    )
    selection = storage_variables or var_name
    probe.close()

    ds = get_dataset_from_path(
        data_path,
        var_name=selection,
        file_name=file_name,
        normalize_latlon_coords=True,
        use_mfdataset=True,
        chunking=True,
        available_mem_gib=max_memory_gib,
        chunk_type=ChunkType.TIME,
        force_ascending_y=True,
    )
    ds = normalize_time(ds)
    if input_format == "mhm_states":
        da = calculate_total_water_storage(ds, storage_variables, scale_factor, label)
    else:
        name = var_name or get_single_data_var(ds)
        if name not in ds.data_vars:
            msg = (
                f"Variable '{name}' not found in the {label} dataset. "
                f"Available data variables: {list(ds.data_vars)}."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        da = convert_water_storage_to_mm(
            ds[name], scale_factor=scale_factor, label=label
        )

    # selecting one variable drops the CF bounds variable, which is the only
    # place the record states the window each of its values covers
    time_bounds = get_time_bounds(ds)
    if time_bounds is not None:
        logger.info(f"The {label} record states the time window of every step.")
        da = set_time_bounds(da, time_bounds)

    da = normalize_lat_lon(da).squeeze(drop=True)
    missing_dims = {"time", "lat", "lon"} - set(da.dims)
    if missing_dims:
        msg = f"The {label} record is missing the dimensions {missing_dims}."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return da.transpose("time", "lat", "lon", ...), input_format


# ---------------------------------------------------------------------------
# grid and calendar normalization
# ---------------------------------------------------------------------------


def validate_shared_grid(input_da, ref_da):
    """Fail when two fields do not share identical lat and lon coordinates.

    Every later step compares the two records by position, so a grid that only
    almost matches would produce a plausible but wrong map.

    Args:
        input_da: Input DataArray with ``lat`` and ``lon``.
        ref_da: Reference DataArray with ``lat`` and ``lon``.

    Returns
    -------
        None. Raises ``ValueError`` when the grids differ.
    """
    for axis in ("lat", "lon"):
        if not np.array_equal(input_da[axis].values, ref_da[axis].values):
            msg = (
                f"The input and reference {axis} coordinates differ after "
                f"regridding ({input_da.sizes[axis]} against "
                f"{ref_da.sizes[axis]} cells). They must match before the "
                f"per cell comparison."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)


def resample_twsa_to_reference_calendar(input_da, ref_da):
    """Bring both records onto the calendar of the reference.

    The reference keeps every one of its steps, and the input is averaged over
    the window each reference value covers. A reference that states its windows
    through CF time bounds is taken at its word, which is what a satellite
    product needs whose steps drift against the calendar or leave gaps. One
    without bounds has its windows derived from its stamps.

    Args:
        input_da: Input DataArray with a ``time`` dimension.
        ref_da: Reference DataArray with a ``time`` dimension.

    Returns
    -------
        Tuple of the two DataArrays on the shared calendar.
    """
    input_da, ref_da = resample_to_reference_windows(input_da, ref_da)
    logger.info(f"Both datasets share {ref_da.sizes['time']} reference steps.")
    return input_da, ref_da


def create_baseline_slice(baseline_start_year, baseline_end_year):
    """Create the time slice of the anomaly baseline period.

    Args:
        baseline_start_year: First calendar year of the baseline.
        baseline_end_year: Last calendar year of the baseline, inclusive.

    Returns
    -------
        The ``slice`` covering both years completely.
    """
    if baseline_end_year < baseline_start_year:
        msg = (
            f"The baseline period {baseline_start_year} to {baseline_end_year} "
            f"ends before it starts."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return slice(f"{baseline_start_year}-01-01", f"{baseline_end_year}-12-31")


def create_twsa_fields(
    input_da, input_format, ref_da, ref_format, baseline_slice, ref_name
):
    """Turn both monthly records into anomalies against the baseline mean.

    A record that already holds an anomaly is left as it is, because its own
    baseline was removed before the tool saw it.

    Args:
        input_da: Monthly input DataArray.
        input_format: Resolved input format.
        ref_da: Monthly reference DataArray.
        ref_format: Resolved reference format.
        baseline_slice: ``slice`` of the baseline period.
        ref_name: Reference name used in the warning message.

    Returns
    -------
        Tuple of (input anomaly, reference anomaly, input baseline mean,
        reference baseline mean). A baseline mean is None for a record that
        already arrived as an anomaly.
    """
    input_baseline = None
    ref_baseline = None
    if input_format == "twsa":
        logger.info("The input already holds an anomaly, keeping it as it is.")
        input_twsa = input_da
    else:
        input_twsa, input_baseline = calculate_anomaly(
            input_da, baseline_slice, require_full_coverage=True, label="input"
        )
    if ref_format == "twsa":
        logger.warning(
            f"The reference {ref_name} already holds an anomaly, so its baseline "
            f"period is unknown and assumed to match {baseline_slice.start} to "
            f"{baseline_slice.stop}. A different reference baseline shifts its "
            f"mean and therefore mostly the bias term of the KGE."
        )
        ref_twsa = ref_da
    else:
        ref_twsa, ref_baseline = calculate_anomaly(
            ref_da, baseline_slice, require_full_coverage=True, label="reference"
        )
    return input_twsa, ref_twsa, input_baseline, ref_baseline


def create_kge_baseline_offsets(
    input_baseline, ref_baseline, fallback=DEFAULT_KGE_BASELINE_OFFSET
):
    """Pick the storage level that is added back to each anomaly for the KGE.

    The bias ratio of two anomalies divides two near zero residuals, so it
    explodes on every cell whose reference mean sits close to zero. Adding a
    storage level back never touches the correlation or the variability ratio,
    which are both invariant against a shift, and only makes the bias readable.

    Where both storage levels are known each record gets its own back, which
    restores the record and makes the bias the real storage ratio again. Where
    only one level is known the reference storage cannot be reconstructed, so
    the same offset is added to both: the bias then reduces to about one and
    carries no information, which is the honest answer when the reference never
    revealed the level its anomaly was taken from.

    Args:
        input_baseline: Per cell baseline mean of the input, or None.
        ref_baseline: Per cell baseline mean of the reference, or None.
        fallback: Storage level in mm used when neither baseline is known.

    Returns
    -------
        Tuple of (input offset, reference offset, description of the choice).
    """
    if input_baseline is not None and ref_baseline is not None:
        logger.info(
            "Adding each record its own baseline mean back, so the KGE bias term "
            "is the ratio of the two storages."
        )
        return input_baseline, ref_baseline, "own baseline mean per record"
    known_baseline = ref_baseline if ref_baseline is not None else input_baseline
    if known_baseline is not None:
        missing = "reference" if ref_baseline is None else "input"
        logger.warning(
            f"The {missing} arrived as an anomaly, so its storage level is "
            f"unknown and cannot be restored. The same offset is added to both "
            f"records instead, which leaves the correlation and the variability "
            f"ratio untouched but reduces the bias term to about one, so it no "
            f"longer measures a storage bias."
        )
        return known_baseline, known_baseline, "common baseline mean, bias neutral"
    logger.warning(
        f"Both records arrived as anomalies, so no storage level is known. The "
        f"constant {fallback:g} mm is added to both instead, which leaves the "
        f"correlation and the variability ratio untouched but reduces the bias "
        f"term to about one, so it no longer measures a storage bias."
    )
    return fallback, fallback, f"constant {fallback:g} mm, bias neutral"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def calculate_twsa_kge(
    input_twsa, ref_twsa, min_valid_months, input_offset, ref_offset
):
    """Calculate the per cell KGE of two monthly anomaly fields.

    Each record is shifted by its own offset before it is scored, which leaves
    the correlation and the variability ratio untouched and gives the bias ratio
    a storage as its denominator instead of a near zero anomaly residual. See
    `create_kge_baseline_offsets`.

    Args:
        input_twsa: Monthly input anomaly with ``time``, ``lat`` and ``lon``.
        ref_twsa: Monthly reference anomaly on the same grid and time axis.
        min_valid_months: Fewest paired months a cell needs for a value.
        input_offset: Storage level added back to the input, a field or constant.
        ref_offset: Storage level added back to the reference.

    Returns
    -------
        Dataset with the KGE, its components, the per cell counts and the two
        offsets the bias term was taken with.
    """
    # a step missing in either record cannot be compared, so it is dropped from
    # both before any moment is taken
    pairwise_valid = input_twsa.notnull() & ref_twsa.notnull()
    kge_ds = calculate_kling_gupta_efficiency_per_cell(
        input_twsa.where(pairwise_valid) + input_offset,
        ref_twsa.where(pairwise_valid) + ref_offset,
        min_valid_pairs=min_valid_months,
    )
    kge_ds = kge_ds.rename(
        valid_pairs="compared_months",
        observed_mean="ref_mean_twsa",
        observed_std="ref_std_twsa",
    )
    # the moments were taken on the shifted record, so the reported mean is
    # turned back into the anomaly mean and the offsets are written next to it
    kge_ds["ref_mean_twsa"] = kge_ds["ref_mean_twsa"] - ref_offset
    kge_ds["kge_offset_input"] = input_offset
    kge_ds["kge_offset_ref"] = ref_offset
    return kge_ds


def log_kge_coverage(kge_ds, min_valid_months):
    """Log how many cells got a KGE and why the others did not.

    Args:
        kge_ds: Dataset returned by `calculate_twsa_kge`.
        min_valid_months: Threshold used for the comparison.

    Returns
    -------
        The number of cells holding a KGE.
    """
    compared = kge_ds["compared_months"]
    scored = int(kge_ds["kge"].notnull().sum())
    no_data = int((compared == 0).sum())
    too_short = int(((compared > 0) & (compared < min_valid_months)).sum())
    degenerate = int((compared >= min_valid_months).sum()) - scored
    logger.info(
        f"{scored} cells got a KGE. {no_data} hold no compared month, "
        f"{too_short} fewer than the required {min_valid_months}, and "
        f"{degenerate} a constant or zero mean series."
    )
    return scored


def log_zero_mean_beta_risk(kge_ds):
    """Warn about the cells whose near zero bias denominator makes it noise.

    The denominator of the bias ratio is the reference mean plus the baseline
    offset, so a cell only stays unstable when the offset does not lift it away
    from zero either.

    Args:
        kge_ds: Dataset returned by `calculate_twsa_kge`.

    Returns
    -------
        The number of affected cells.
    """
    beta_denominator = kge_ds["ref_mean_twsa"] + kge_ds["kge_offset_ref"]
    unstable = (
        np.abs(beta_denominator) < ZERO_MEAN_BETA_TOLERANCE * kge_ds["ref_std_twsa"]
    ) & kge_ds["kge"].notnull()
    affected = int(unstable.sum())
    scored = int(kge_ds["kge"].notnull().sum())
    if affected:
        share = 100 * affected / scored if scored else 0
        logger.warning(
            f"{affected} of {scored} scored cells ({share:.1f} %) have a "
            f"reference mean plus baseline offset below "
            f"{ZERO_MEAN_BETA_TOLERANCE:.0%} of their own spread. Their bias "
            f"ratio, and with it their KGE, is not interpretable. Read alpha "
            f"and gamma for those cells."
        )
    return affected


def set_metric_attributes(metric_ds, run_attributes):
    """Add CF attributes and the run settings to the metric dataset.

    Args:
        metric_ds: Dataset holding the calculated metrics.
        run_attributes: Settings recorded on the dataset.

    Returns
    -------
        The dataset with attributes set.
    """
    descriptions = {
        "kge": ("Kling-Gupta efficiency", "1"),
        "alpha": ("ratio of the standard deviations", "1"),
        "beta": ("ratio of the means shifted by the baseline offset", "1"),
        "gamma": ("correlation of the two anomalies", "1"),
        "compared_months": ("months compared in this cell", "1"),
        "ref_mean_twsa": ("mean reference anomaly", "mm"),
        "ref_std_twsa": ("standard deviation of the reference anomaly", "mm"),
        "rmse": ("root mean square error of the two anomalies", "mm"),
        "kge_offset_input": (
            "storage level added to the input anomaly before scoring",
            "mm",
        ),
        "kge_offset_ref": (
            "storage level added to the reference anomaly before scoring",
            "mm",
        ),
    }
    for name, (long_name, units) in descriptions.items():
        if name in metric_ds:
            metric_ds[name].attrs = {"long_name": long_name, "units": units}
    metric_ds.attrs.update(run_attributes)
    return metric_ds


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_twsa_dataset(ds, file_path):
    """Write a TWSA evaluation dataset to a compressed NetCDF file.

    Args:
        ds: Dataset to write.
        file_path: Target file path.

    Returns
    -------
        The written file path.
    """
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    logger.info(f"Writing {file_path}")
    write_xarray_to_file(ds, file_path, encoding=encoding)
    return Path(file_path)


def get_metric_plot_style(metric_name, kge_vmin=KGE_VMIN):
    """Return the colormap and colour limits of one metric.

    Args:
        metric_name: One of "kge", "alpha", "beta", "gamma" or "rmse".
        kge_vmin: Lower end of the KGE colour scale.

    Returns
    -------
        Tuple of (colormap name, vmin, vmax), where an upper limit of None
        leaves the scale to the data.
    """
    if metric_name == "kge":
        return "viridis", kge_vmin, KGE_VMAX
    if metric_name == "rmse":
        # an error is best at zero and has no natural upper end
        return "magma_r", 0.0, None
    if metric_name == "gamma":
        return "RdBu_r", -1.0, 1.0
    # a ratio is good at one, so it is shown on a scale centred there
    return "RdBu_r", 0.0, 2.0


def create_metric_maps(
    metric_ds,
    output_dir,
    input_name,
    ref_name,
    kge_vmin=KGE_VMIN,
    region_name=None,
    metrics=("kge", *KGE_COMPONENTS),
):
    """Plot the selected metrics and the KGE components as maps.

    Args:
        metric_ds: Dataset holding the calculated metrics.
        output_dir: Directory the maps are written to.
        input_name: Input dataset name used in the title and file name.
        ref_name: Reference dataset name used in the title and file name.
        kge_vmin: Lower end of the KGE colour scale.
        region_name: Region the map is limited to, or None for the domain.
        metrics: Metric names to plot.

    Returns
    -------
        Dict of the written file paths per metric.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{sanitize_name(input_name)}_vs_{sanitize_name(ref_name)}"
    limits = {}
    if region_name:
        bounds = WMO_REGION_BOUNDS[region_name]
        limits = {
            "x_min": bounds["lon_slice"].start,
            "x_max": bounds["lon_slice"].stop,
            "y_min": bounds["lat_slice"].start,
            "y_max": bounds["lat_slice"].stop,
        }
        suffix = f"{suffix}_region_{sanitize_name(region_name)}"

    written = {}
    for metric in metrics:
        if metric not in metric_ds or not bool(metric_ds[metric].notnull().any()):
            logger.info(f"Skipping the {metric} map, it holds no valid cell.")
            continue
        cmap, vmin, vmax = get_metric_plot_style(metric, kge_vmin)
        label = metric.upper() if metric in AVAILABLE_METRICS else metric
        output_file = output_dir / f"twsa_metric_{metric}_{suffix}.png"
        plot_map(
            metric_ds[metric],
            cb_label=f"{label} [mm]" if metric == "rmse" else label,
            title=(
                f"TWSA {label}: "
                f"{input_name} vs {ref_name}{format_region_title(region_name)}"
            ),
            out_path=output_file,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            under_color=KGE_UNDER_COLOR if metric == "kge" else None,
            **limits,
        )
        key = f"{metric}_map"
        written[key] = output_file
    return written


def create_cell_line_segments(da, max_cell_lines):
    """Build the per cell line segments of a spaghetti plot.

    Cells are thinned deterministically by taking every k-th valid cell, so the
    same input always draws the same lines.

    Args:
        da: DataArray with ``time``, ``lat`` and ``lon``.
        max_cell_lines: Most lines to build, 0 for every valid cell.

    Returns
    -------
        Tuple of (list of (time, value) arrays, drawn count, valid count).
    """
    times = mdates.date2num(pd.DatetimeIndex(da["time"].values).to_pydatetime())
    values = da.values.reshape(da.sizes["time"], -1)
    valid_columns = np.flatnonzero(np.isfinite(values).any(axis=0))
    step = 1
    if max_cell_lines and valid_columns.size > max_cell_lines:
        step = int(np.ceil(valid_columns.size / max_cell_lines))
    drawn_columns = valid_columns[::step]
    segments = [
        np.column_stack((times, values[:, column]))[np.isfinite(values[:, column])]
        for column in drawn_columns
    ]
    return segments, len(segments), int(valid_columns.size)


def calculate_latitude_weighted_mean(da):
    """Average a field over lat and lon, weighting cells by cos(latitude).

    An unweighted mean over a geographic grid over-weights the small cells at
    high latitudes.

    Args:
        da: DataArray with ``lat`` and ``lon`` dimensions.

    Returns
    -------
        The weighted mean with the spatial dimensions reduced away.
    """
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights.fillna(0)).mean(("lat", "lon"))


def create_region_cell_series_plot(
    input_twsa,
    ref_twsa,
    output_file,
    input_name,
    ref_name,
    region_name=None,
    max_cell_lines=DEFAULT_MAX_REGION_CELL_LINES,
):
    """Plot every grid cell over time plus the region mean, for both datasets.

    Args:
        input_twsa: Monthly input anomaly of the region.
        ref_twsa: Monthly reference anomaly of the region.
        output_file: Target file path.
        input_name: Input dataset name used in the legend.
        ref_name: Reference dataset name used in the legend.
        region_name: Region shown, or None for the whole domain.
        max_cell_lines: Most cell lines drawn per dataset, 0 for all.

    Returns
    -------
        The written file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    captions = []
    for da, color, name in (
        (input_twsa, INPUT_COLOR, input_name),
        (ref_twsa, REF_COLOR, ref_name),
    ):
        segments, drawn, valid = create_cell_line_segments(da, max_cell_lines)
        if segments:
            # one artist for thousands of lines, rasterized so the file stays small
            opacity = max(0.02, min(0.15, 20 / drawn))
            collection = LineCollection(
                segments, colors=color, linewidths=CELL_LINE_WIDTH, alpha=opacity
            )
            collection.set_rasterized(True)
            ax.add_collection(collection)
        captions.append(f"{name}: {drawn} of {valid} cells")
        region_mean = calculate_latitude_weighted_mean(da)
        ax.plot(
            pd.DatetimeIndex(da["time"].values),
            region_mean.values,
            color=color,
            linewidth=REGION_MEAN_LINE_WIDTH,
            label=f"{name} mean",
            zorder=5,
        )

    ax.axhline(0.0, color=CAPTION_COLOR, linewidth=0.8, zorder=1)
    style_axes(ax, show_grid=True)
    ax.set_ylabel("TWS anomaly [mm]")
    ax.set_title(
        f"TWS anomaly per cell: {input_name} vs {ref_name}"
        f"{format_region_title(region_name)}",
        color=CAPTION_COLOR,
    )
    time_index = pd.DatetimeIndex(input_twsa["time"].values)
    span_years = max(1, (time_index[-1] - time_index[0]).days // 365)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, -(-span_years // 15))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(
        time_index[0] - pd.Timedelta(days=45), time_index[-1] + pd.Timedelta(days=45)
    )
    ax.annotate(
        "; ".join(captions),
        xy=(0.0, -0.18),
        xycoords="axes fraction",
        color=CAPTION_COLOR,
        fontsize=8,
    )
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


# ---------------------------------------------------------------------------
# regions and main worker
# ---------------------------------------------------------------------------


def create_region_outputs(
    input_twsa,
    ref_twsa,
    metric_ds,
    output_dir,
    region_names,
    input_name,
    ref_name,
    kge_vmin=KGE_VMIN,
    max_cell_lines=DEFAULT_MAX_REGION_CELL_LINES,
    metrics=("kge",),
):
    """Write the cell series plot and the metric maps for every region with data.

    Args:
        input_twsa: Monthly input anomaly of the whole domain.
        ref_twsa: Monthly reference anomaly of the whole domain.
        metric_ds: Dataset holding the calculated metrics, or None for none.
        output_dir: Directory the outputs are written to.
        region_names: Regions to write.
        input_name: Input dataset name.
        ref_name: Reference dataset name.
        kge_vmin: Lower end of the KGE colour scale.
        max_cell_lines: Most cell lines drawn per dataset.
        metrics: Metric names mapped per region, empty for the series plot only.

    Returns
    -------
        Dict of the written output file paths per region.
    """
    output_dir = Path(output_dir)
    written = {}
    for region_name in region_names:
        region_input = crop_to_region(input_twsa, region_name)
        region_ref = crop_to_region(ref_twsa, region_name)
        if region_input.sizes["lat"] == 0 or region_input.sizes["lon"] == 0:
            logger.info(f"Skipping region {region_name}: no cells inside its bounds.")
            continue
        if not bool(region_input.notnull().any()) or not bool(
            region_ref.notnull().any()
        ):
            logger.info(f"Skipping region {region_name}: no valid data inside it.")
            continue
        suffix = (
            f"{sanitize_name(input_name)}_vs_{sanitize_name(ref_name)}"
            f"_region_{sanitize_name(region_name)}"
        )
        written[f"{region_name} cell_series_plot"] = create_region_cell_series_plot(
            region_input,
            region_ref,
            output_dir / f"twsa_cells_{suffix}.png",
            input_name,
            ref_name,
            region_name=region_name,
            max_cell_lines=max_cell_lines,
        )
        if metric_ds is None or not metrics:
            continue
        region_maps = create_metric_maps(
            metric_ds,
            output_dir,
            input_name,
            ref_name,
            kge_vmin=kge_vmin,
            region_name=region_name,
            metrics=metrics,
        )
        for key, path in region_maps.items():
            written[f"{region_name} {key}"] = path
    return written


def twsa_evaluation(  # noqa: PLR0913
    input_path,
    ref_path,
    output_dir,
    input_file_name="*.nc",
    ref_file_name="*.nc",
    input_var=None,
    ref_var=None,
    input_name="mhm",
    ref_name="ref",
    input_format=AUTO_FORMAT,
    ref_format=AUTO_FORMAT,
    input_scale=None,
    ref_scale=None,
    baseline_start_year=DEFAULT_BASELINE_START_YEAR,
    baseline_end_year=DEFAULT_BASELINE_END_YEAR,
    min_valid_months=DEFAULT_MIN_VALID_MONTHS,
    regions="all",
    max_region_cell_lines=DEFAULT_MAX_REGION_CELL_LINES,
    kge_vmin=KGE_VMIN,
    metrics=AVAILABLE_METRICS,
    write_twsa=True,
    max_memory_gib=8.0,
):
    """Evaluate a modelled total water storage anomaly against a reference.

    Args:
        input_path: Path to the model file or a directory searched recursively.
        ref_path: Path to the reference file or a directory searched recursively.
        output_dir: Directory the outputs are written to.
        input_file_name: Glob pattern for the recursive model search.
        ref_file_name: Glob pattern for the recursive reference search.
        input_var: Model variable, or None to detect it.
        ref_var: Reference variable, or None to detect it.
        input_name: Model dataset name used in titles and file names.
        ref_name: Reference dataset name used in titles and file names.
        input_format: One of `INPUT_FORMATS`, or "auto".
        ref_format: One of `INPUT_FORMATS`, or "auto".
        input_scale: Millimetres per model data unit, or None to use the units.
        ref_scale: Millimetres per reference data unit, or None.
        baseline_start_year: First calendar year of the anomaly baseline.
        baseline_end_year: Last calendar year of the baseline, inclusive.
        min_valid_months: Fewest paired months a cell needs for a metric.
        regions: "all", "none" or a comma separated list of region names.
        max_region_cell_lines: Most cell lines drawn per dataset in a region plot.
        kge_vmin: Lower end of the KGE colour scale.
        metrics: Names out of `AVAILABLE_METRICS` that every cell is scored
            with, all of them by default. A metric that is not named is neither
            calculated, written nor plotted.
        write_twsa: Also write the normalized monthly anomaly fields.
        max_memory_gib: Memory budget the time chunks are sized against.

    Returns
    -------
        Dict of the written output file paths.
    """
    output_dir = Path(output_dir)
    # check both selections first, so a typo fails before the data is read
    requested_metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = [name for name in requested_metrics if name not in AVAILABLE_METRICS]
    if unknown:
        msg = (
            f"Unknown metric(s) {', '.join(unknown)}. "
            f"Available metrics: {', '.join(AVAILABLE_METRICS)}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    selected_metrics = [name for name in AVAILABLE_METRICS if name in requested_metrics]
    if not selected_metrics and not write_twsa:
        msg = (
            "No metric is selected and the anomaly fields are switched off, so "
            "there is nothing to calculate. Select a metric or keep the fields."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    region_names = select_regions(regions)
    baseline_slice = create_baseline_slice(baseline_start_year, baseline_end_year)

    input_da, input_format = read_twsa_data_array(
        input_path,
        input_file_name,
        input_var,
        input_format,
        input_scale,
        "input",
        max_memory_gib,
    )
    ref_da, ref_format = read_twsa_data_array(
        ref_path,
        ref_file_name,
        ref_var,
        ref_format,
        ref_scale,
        "reference",
        max_memory_gib,
    )

    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)
    input_da, ref_da = regrid_to_coarser_grid(input_da, ref_da)
    # the aggregation can shift the outer edges, so the overlap is taken again
    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)
    validate_shared_grid(input_da, ref_da)

    time_slice = get_overlapping_time_slice(input_da, ref_da)
    input_da = input_da.sel(time=time_slice)
    ref_da = ref_da.sel(time=time_slice)
    input_da, ref_da = resample_twsa_to_reference_calendar(input_da, ref_da)

    input_twsa, ref_twsa, input_baseline, ref_baseline = create_twsa_fields(
        input_da, input_format, ref_da, ref_format, baseline_slice, ref_name
    )
    del input_da, ref_da
    # Every step below reads these again: the KGE, both anomaly files and every
    # region plot. A lazy field is rebuilt from its whole read, regrid and
    # window graph each time, so it is computed once here instead.
    input_twsa = materialize_if_within_budget(
        input_twsa, max_memory_gib, f"the {input_name} anomaly"
    )
    ref_twsa = materialize_if_within_budget(
        ref_twsa, max_memory_gib, f"the {ref_name} anomaly"
    )

    metric_ds = None
    offset_source = None
    if "kge" in selected_metrics:
        input_offset, ref_offset, offset_source = create_kge_baseline_offsets(
            input_baseline, ref_baseline
        )
        logger.info("Scoring every grid cell with the Kling-Gupta efficiency.")
        metric_ds = calculate_twsa_kge(
            input_twsa, ref_twsa, min_valid_months, input_offset, ref_offset
        )
    if "rmse" in selected_metrics:
        logger.info("Scoring every grid cell with the root mean square error.")
        # the error is taken on the anomalies themselves, where the offsets the
        # KGE bias needs would only shift the two records apart
        rmse = calculate_root_mean_square_error_per_cell(
            input_twsa, ref_twsa, min_valid_pairs=min_valid_months
        )
        if metric_ds is None:
            metric_ds = rmse.rename("rmse").to_dataset()
            # the KGE writes the count next to its score, so a run without it
            # still states how many months a cell was scored on
            metric_ds["compared_months"] = (
                input_twsa.notnull() & ref_twsa.notnull()
            ).sum("time")
        else:
            metric_ds["rmse"] = rmse
    if metric_ds is not None:
        metric_ds = metric_ds.compute()
    if "kge" in selected_metrics:
        log_kge_coverage(metric_ds, min_valid_months)
        log_zero_mean_beta_risk(metric_ds)

    run_attributes = {
        "input_dataset": input_name,
        "reference_dataset": ref_name,
        "input_format": input_format,
        "reference_format": ref_format,
        "reference_calendar": "reference time windows",
        "baseline_period": f"{baseline_start_year}-{baseline_end_year}",
        "min_valid_months": min_valid_months,
        "compared_months_total": int(input_twsa.sizes["time"]),
        "metrics": ", ".join(selected_metrics) or "none",
    }
    if offset_source is not None:
        run_attributes["kge_offset_source"] = offset_source

    suffix = f"{sanitize_name(input_name)}_vs_{sanitize_name(ref_name)}"
    written_files = {}
    if metric_ds is not None:
        metric_ds = set_metric_attributes(metric_ds, run_attributes)
        written_files["metrics"] = write_twsa_dataset(
            metric_ds, output_dir / f"twsa_metrics_{suffix}.nc"
        )
        # a KGE is only readable next to its components, so they are mapped with it
        map_metrics = []
        for metric in selected_metrics:
            map_metrics.append(metric)
            if metric == "kge":
                map_metrics.extend(KGE_COMPONENTS)
        logger.info(f"Plotting {', '.join(map_metrics)} as maps.")
        written_files.update(
            create_metric_maps(
                metric_ds,
                output_dir,
                input_name,
                ref_name,
                kge_vmin=kge_vmin,
                metrics=map_metrics,
            )
        )
    if write_twsa:
        for da, name, key in (
            (input_twsa, input_name, "input_twsa"),
            (ref_twsa, ref_name, "ref_twsa"),
        ):
            twsa_ds = da.rename("twsa").to_dataset()
            twsa_ds["twsa"].attrs = {
                "long_name": "total water storage anomaly",
                "units": "mm",
            }
            twsa_ds.attrs.update({**run_attributes, "dataset": name})
            written_files[key] = write_twsa_dataset(
                twsa_ds, output_dir / f"twsa_monthly_{sanitize_name(name)}.nc"
            )
    if region_names:
        logger.info(f"Writing the outputs of {len(region_names)} region(s).")
    written_files.update(
        create_region_outputs(
            input_twsa,
            ref_twsa,
            metric_ds,
            output_dir,
            region_names,
            input_name,
            ref_name,
            kge_vmin=kge_vmin,
            max_cell_lines=max_region_cell_lines,
            metrics=("kge",) if "kge" in selected_metrics else (),
        )
    )
    return written_files
