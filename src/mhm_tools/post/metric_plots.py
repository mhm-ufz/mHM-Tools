"""
Create metric plots from metric CSV files.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mhm_tools.common.catchment_maps import (
    calculate_metric_medians,
    find_matching_geometry_file,
    write_catchment_median_maps,
)
from mhm_tools.common.plotter import (
    create_metric_summary_rows,
    plot_metric_cdf_comparison,
    plot_metric_violin_comparison,
    write_metric_plot_overview_pdf,
)

logger = logging.getLogger(__name__)


def _path_has_wildcards(path):
    """Return whether a path string contains glob wildcards.

    Parameters
    ----------
    path : str or Path
        Path or glob pattern to inspect.

    Returns
    -------
    bool
        True if the path contains glob wildcards.
    """
    return any(token in str(path) for token in ("*", "?", "["))


def _iter_wildcard_matches(pattern):
    """Return sorted paths matching a relative or absolute glob pattern.

    Parameters
    ----------
    pattern : str or Path
        Relative or absolute glob pattern.

    Returns
    -------
    list[Path]
        Matching paths sorted by path string.
    """
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        anchor = Path(pattern_path.anchor)
        relative_pattern = str(pattern_path.relative_to(anchor))
        matches = anchor.glob(relative_pattern)
    else:
        matches = Path().glob(str(pattern))
    return sorted(matches, key=lambda path: str(path))


def _unique_paths(paths):
    """Return paths without duplicates while preserving order.

    Parameters
    ----------
    paths : Sequence[str or Path]
        Paths to deduplicate.

    Returns
    -------
    list[Path]
        Deduplicated paths.
    """
    unique = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _none_if_empty_sequence(value):
    """Return None for empty optional sequences.

    Parameters
    ----------
    value : Sequence or None
        Optional sequence value to normalize.

    Returns
    -------
    Sequence or None
        None if value is an empty sequence, otherwise the original value.
    """
    if value is not None and not value:
        return None
    return value


def _get_realisation_output_dir(output_dir, input_name, split_by_input):
    """Get the output directory for one metric plot realisation.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory.
    input_name : str
        Realisation name.
    split_by_input : bool
        Whether to create one subdirectory per realisation.

    Returns
    -------
    Path
        Output directory for the realisation.
    """
    output_dir = Path(output_dir)
    if not split_by_input:
        return output_dir
    safe_name = str(input_name).replace("/", "_").replace("\\", "_")
    return output_dir / safe_name


def _get_safe_output_name(value):
    """Create a filesystem-safe output name part.

    Parameters
    ----------
    value : object
        Value used in an output filename.

    Returns
    -------
    str
        Safe filename part.
    """
    safe_name = str(value).strip().replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace(" ", "_")
    return safe_name or "unknown"


def _create_metric_label_metadata(input_names, name_fields=None, name_separator="_"):
    """Create metadata fields parsed from metric input labels.

    Parameters
    ----------
    input_names : Sequence[str]
        Labels used for metric plot inputs.
    name_fields : Sequence[str], optional
        Field names matching separated label parts.
    name_separator : str, optional
        Separator used to split labels.

    Returns
    -------
    dict[str, dict[str, str]]
        Metadata by input label.
    """
    metadata_by_label = {}
    name_fields = list(name_fields or [])
    for input_name in input_names:
        metadata = {"label": input_name}
        if name_fields:
            parts = str(input_name).split(name_separator)
            if len(parts) == len(name_fields):
                metadata.update(dict(zip(name_fields, parts)))
        metadata_by_label[input_name] = metadata
    return metadata_by_label


def _get_metric_group_value(label_metadata, field_name):
    """Get a group value from label metadata.

    Parameters
    ----------
    label_metadata : Mapping[str, str]
        Parsed label metadata.
    field_name : str
        Metadata field to read.

    Returns
    -------
    str or None
        Group value, or None for reference labels.
    """
    return label_metadata.get(field_name)


def _get_metric_plot_values_by_group(values_by_input, label_metadata, group_field):
    """Group metric values by one metadata field.

    Parameters
    ----------
    values_by_input : Mapping[str, Sequence[float]]
        Metric values by input label.
    label_metadata : Mapping[str, Mapping[str, str]]
        Parsed label metadata by input label.
    group_field : str
        Metadata field used for grouping.

    Returns
    -------
    dict[str, dict[str, Sequence[float]]]
        Metric values by group value and input label.
    """
    grouped_values = {}
    reference_values = {}
    for label, values in values_by_input.items():
        group_value = _get_metric_group_value(
            label_metadata.get(label, {}), group_field
        )
        if group_value is None:
            reference_values[label] = values
            continue
        grouped_values.setdefault(group_value, {})[label] = values
    for _group_value, group_values in grouped_values.items():
        group_values.update(reference_values)
    if not grouped_values and reference_values:
        grouped_values["ungrouped"] = reference_values
    return grouped_values


def _get_distinct_metric_plot_values(input_names, label_metadata, field_name):
    """Get distinct metadata values in input label order.

    Parameters
    ----------
    input_names : Sequence[str]
        Input labels in plot order.
    label_metadata : Mapping[str, Mapping[str, str]]
        Parsed label metadata by input label.
    field_name : str
        Metadata field to collect.

    Returns
    -------
    list[str]
        Distinct metadata values.
    """
    values = []
    for input_name in input_names:
        field_value = label_metadata.get(input_name, {}).get(field_name)
        if field_value is None or field_value in values:
            continue
        values.append(field_value)
    return values


def _get_metric_plot_colors_by_label(input_names, label_metadata, color_by):
    """Create metric plot colors from a metadata field.

    Parameters
    ----------
    input_names : Sequence[str]
        Input labels in plot order.
    label_metadata : Mapping[str, Mapping[str, str]]
        Parsed label metadata by input label.
    color_by : str or None
        Metadata field used for colors.

    Returns
    -------
    dict[str, object] or None
        Colors by input label.
    """
    if color_by is None:
        return None
    color_values = _get_distinct_metric_plot_values(
        input_names, label_metadata, color_by
    )
    tab20_colors = plt.get_cmap("tab20").colors
    continuous_cmap = plt.get_cmap("nipy_spectral")
    colors_by_value = {}
    for color_index, color_value in enumerate(color_values):
        if len(color_values) <= len(tab20_colors):
            colors_by_value[color_value] = tab20_colors[color_index]
        else:
            color_fraction = 0.05 + (0.9 * color_index / max(len(color_values) - 1, 1))
            colors_by_value[color_value] = continuous_cmap(color_fraction)
    colors_by_label = {}
    for input_name in input_names:
        color_value = label_metadata.get(input_name, {}).get(color_by)
        if color_value is None:
            colors_by_label[input_name] = "black"
        else:
            colors_by_label[input_name] = colors_by_value[color_value]
    return colors_by_label


def _get_metric_plot_linestyles_by_label(input_names, label_metadata, style_by):
    """Create metric plot line styles from a metadata field.

    Parameters
    ----------
    input_names : Sequence[str]
        Input labels in plot order.
    label_metadata : Mapping[str, Mapping[str, str]]
        Parsed label metadata by input label.
    style_by : str or None
        Metadata field used for line styles.

    Returns
    -------
    dict[str, object] or None
        Line styles by input label.
    """
    if style_by is None:
        return None
    style_values = _get_distinct_metric_plot_values(
        input_names, label_metadata, style_by
    )
    style_cycle = [
        "-",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (1, 1)),
        (0, (5, 2, 1, 2)),
    ]
    styles_by_value = {
        style_value: style_cycle[style_index % len(style_cycle)]
        for style_index, style_value in enumerate(style_values)
    }
    styles_by_label = {}
    for input_name in input_names:
        style_value = label_metadata.get(input_name, {}).get(style_by)
        styles_by_label[input_name] = styles_by_value.get(style_value, "-")
    return styles_by_label


def _write_grouped_metric_plots(
    values_by_variable,
    output_dir,
    output_prefix,
    plot_types,
    group_by,
    label_metadata,
    colors_by_label=None,
    linestyles_by_label=None,
    dpi=450,
):
    """Write additional metric plots grouped by metadata fields.

    Parameters
    ----------
    values_by_variable : Mapping[str, Mapping[str, Sequence[float]]]
        Metric values grouped by variable and input label.
    output_dir : str or Path
        Directory for PNG output files.
    output_prefix : str
        Prefix for CDF output PNG names.
    plot_types : Sequence[str]
        Plot types to write.
    group_by : Sequence[str]
        Metadata fields used for grouped plots.
    label_metadata : Mapping[str, Mapping[str, str]]
        Parsed label metadata by input label.
    colors_by_label : Mapping[str, object], optional
        Colors by input label.
    linestyles_by_label : Mapping[str, object], optional
        CDF line styles by input label.
    dpi : int, optional
        Output image resolution.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    output_files = []
    output_dir = Path(output_dir)
    for group_field in group_by or []:
        safe_group_field = _get_safe_output_name(group_field)
        for variable, values_by_input in values_by_variable.items():
            grouped_values = _get_metric_plot_values_by_group(
                values_by_input=values_by_input,
                label_metadata=label_metadata,
                group_field=group_field,
            )
            for group_value, group_values in grouped_values.items():
                safe_group_value = _get_safe_output_name(group_value)
                if "cdf" in plot_types:
                    output_file = (
                        output_dir / f"{output_prefix}_{variable}_by_"
                        f"{safe_group_field}_{safe_group_value}.png"
                    )
                    plot_metric_cdf_comparison(
                        values_by_label=group_values,
                        variable_name=variable,
                        output_file=output_file,
                        title=f"CDF of {variable}: {group_field}={group_value}",
                        dpi=dpi,
                        colors=colors_by_label,
                        linestyles=linestyles_by_label,
                        show_median_line=True,
                    )
                    output_files.append(output_file)
                    logger.info(f"Wrote grouped CDF plot to {output_file}")
                if "violin" in plot_types:
                    output_file = (
                        output_dir / f"violin_{variable}_by_"
                        f"{safe_group_field}_{safe_group_value}.png"
                    )
                    plot_metric_violin_comparison(
                        values_by_label=group_values,
                        variable_name=variable,
                        output_file=output_file,
                        title=(
                            f"Distribution of {variable}: "
                            f"{group_field}={group_value}"
                        ),
                        dpi=dpi,
                        colors=colors_by_label,
                    )
                    output_files.append(output_file)
                    logger.info(f"Wrote grouped violin plot to {output_file}")
    return output_files


def get_metric_csv_files(input_path, file_names="*.csv"):
    """Get metric CSV files from a file or recursive directory search.

    Parameters
    ----------
    input_path : str or Path
        CSV file, directory containing CSV files, or glob pattern.
    file_names : str, optional
        Glob pattern used when input_path is a directory.

    Returns
    -------
    list[Path]
        Matching CSV files.
    """
    if _path_has_wildcards(input_path):
        matches = _iter_wildcard_matches(input_path)
        if not matches:
            msg = f"No metric input paths match pattern {input_path}"
            raise FileNotFoundError(msg)

        csv_files = []
        for match in matches:
            if match.is_file():
                csv_files.append(match)
            elif match.is_dir():
                csv_files.extend(sorted(match.rglob(file_names)))
        csv_files = _unique_paths(csv_files)
        if csv_files:
            return csv_files
        msg = (
            f"No CSV files matching {file_names!r} found below paths matching "
            f"{input_path}"
        )
        raise FileNotFoundError(msg)

    metric_path = Path(input_path)
    if metric_path.is_file():
        return [metric_path]
    if metric_path.is_dir():
        csv_files = sorted(metric_path.rglob(file_names))
        if csv_files:
            return csv_files
        msg = f"No CSV files matching {file_names!r} found below {metric_path}"
        raise FileNotFoundError(msg)
    msg = f"Metric input path {metric_path} does not exist."
    raise FileNotFoundError(msg)


def _get_wildcard_input_name(input_path):
    """Create a default label for a wildcard metric input path.

    Parameters
    ----------
    input_path : str or Path
        Wildcard input path.

    Returns
    -------
    str
        Default label derived from the last stable path component.
    """
    path = Path(input_path)
    stable_parts = [
        part
        for part in path.parts
        if part != path.anchor and not _path_has_wildcards(part)
    ]
    if stable_parts:
        return stable_parts[-1]
    return str(input_path)


def get_metric_input_names(input_paths, input_names=None):
    """Get plot labels for input paths.

    Parameters
    ----------
    input_paths : Sequence[str or Path]
        CSV files or directories.
    input_names : Sequence[str], optional
        Explicit names for the input paths.

    Returns
    -------
    list[str]
        Plot labels matching input_paths.
    """
    if input_names is not None:
        if len(input_names) != len(input_paths):
            msg = (
                f"--input-names contains {len(input_names)} values but "
                f"--input-paths contains {len(input_paths)} values."
            )
            raise ValueError(msg)
        return list(input_names)

    names = []
    for input_path in input_paths:
        if _path_has_wildcards(input_path):
            names.append(_get_wildcard_input_name(input_path))
            continue
        path = Path(input_path)
        if path.is_file():
            names.append(path.stem)
        else:
            names.append(path.name or str(path))
    return names


def read_metric_values_from_csv_files(csv_files, variables):
    """Read numeric metric values from CSV files.

    Parameters
    ----------
    csv_files : Sequence[Path]
        CSV files to read.
    variables : Sequence[str]
        Exact metric column names to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Numeric finite values grouped by variable name.
    """
    values_by_variable = {variable: [] for variable in variables}
    for csv_file in csv_files:
        metric_df = pd.read_csv(csv_file)
        for variable in variables:
            if variable not in metric_df.columns:
                logger.warning(f"Column {variable!r} missing in {csv_file}")
                continue
            values = pd.to_numeric(metric_df[variable], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            values_by_variable[variable].extend(values.to_numpy(dtype=float))

    return {
        variable: np.asarray(values, dtype=float)
        for variable, values in values_by_variable.items()
    }


def get_metric_values_by_input(
    input_paths,
    variables,
    input_names=None,
    file_names="*.csv",
):
    """Get metric values grouped by variable and input name.

    Parameters
    ----------
    input_paths : Sequence[str or Path]
        CSV files or directories.
    variables : Sequence[str]
        Exact metric column names to extract.
    input_names : Sequence[str], optional
        Names used as plot labels.
    file_names : str, optional
        Glob pattern used for recursive directory searches.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Metric values grouped by variable and input name.
    """
    names = get_metric_input_names(input_paths, input_names)
    values_by_variable = {variable: {} for variable in variables}
    for input_path, input_name in zip(input_paths, names):
        csv_files = get_metric_csv_files(input_path, file_names=file_names)
        logger.info(f"Reading {len(csv_files)} CSV files for {input_name}")
        input_values = read_metric_values_from_csv_files(csv_files, variables)
        for variable, values in input_values.items():
            if values.size == 0:
                logger.warning(f"No finite values for {variable!r} in {input_path}")
                continue
            values_by_variable[variable][input_name] = values

    for variable, values_by_input in values_by_variable.items():
        if not values_by_input:
            msg = f"No finite values found for variable {variable!r}"
            raise ValueError(msg)
    return values_by_variable


def write_metric_plots(  # noqa: PLR0913
    input_paths: Sequence[str],
    variables: Sequence[str],
    output_dir,
    input_names: Optional[Sequence[str]] = None,
    file_names: str = "*.csv",
    output_prefix: str = "cdf",
    plot_types: Optional[Sequence[str]] = None,
    dpi: int = 450,
    shape_paths: Optional[Sequence[str]] = None,
    shape_folder=None,
    mask_paths: Optional[Sequence[str]] = None,
    mask_folder=None,
    mask_var=None,
    geometry_match_mode="auto",
    name_fields: Optional[Sequence[str]] = None,
    name_separator: str = "_",
    group_by: Optional[Sequence[str]] = None,
    color_by: Optional[str] = None,
    style_by: Optional[str] = None,
) -> List[Path]:
    """Write metric comparison plots from metric CSV files.

    Parameters
    ----------
    input_paths : Sequence[str]
        CSV files or directories.
    variables : Sequence[str]
        Exact metric column names to plot.
    output_dir : str or Path
        Directory for PNG output files.
    input_names : Sequence[str], optional
        Names used as plot labels.
    file_names : str, optional
        Glob pattern used for recursive directory searches.
    output_prefix : str, optional
        Prefix for CDF output PNG names.
    plot_types : Sequence[str], optional
        Plot types to write.
    dpi : int, optional
        Output image resolution.
    shape_paths : Sequence[str], optional
        Shapefiles matched one-to-one with input_paths.
    shape_folder : str or Path, optional
        Folder with shapefiles matched by ID.
    mask_paths : Sequence[str], optional
        NetCDF masks matched one-to-one with input_paths.
    mask_folder : str or Path, optional
        Folder with NetCDF masks matched by ID.
    mask_var : str, optional
        Mask variable name.
    geometry_match_mode : str, optional
        Geometry matching mode. Only "auto" is currently supported.
    name_fields : Sequence[str], optional
        Metadata field names parsed from input_names.
    name_separator : str, optional
        Separator used to split input_names into metadata fields.
    group_by : Sequence[str], optional
        Metadata fields used for additional grouped plots.
    color_by : str, optional
        Metadata field used for plot colors.
    style_by : str, optional
        Metadata field used for CDF line styles.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    if plot_types is None:
        plot_types = ["cdf", "violin"]
    plot_types = list(plot_types)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_names = get_metric_input_names(input_paths, input_names)
    label_metadata = _create_metric_label_metadata(
        input_names=input_names,
        name_fields=name_fields,
        name_separator=name_separator,
    )
    colors_by_label = _get_metric_plot_colors_by_label(
        input_names=input_names,
        label_metadata=label_metadata,
        color_by=color_by,
    )
    linestyles_by_label = _get_metric_plot_linestyles_by_label(
        input_names=input_names,
        label_metadata=label_metadata,
        style_by=style_by,
    )
    values_by_variable = {}
    if "cdf" in plot_types or "violin" in plot_types:
        values_by_variable = get_metric_values_by_input(
            input_paths=input_paths,
            input_names=input_names,
            variables=variables,
            file_names=file_names,
        )

    output_files = []
    if "cdf" in plot_types:
        for variable, values_by_input in values_by_variable.items():
            output_file = output_dir / f"{output_prefix}_{variable}.png"
            plot_metric_cdf_comparison(
                values_by_label=values_by_input,
                variable_name=variable,
                output_file=output_file,
                dpi=dpi,
                colors=colors_by_label,
                linestyles=linestyles_by_label,
                show_median_line=True,
            )
            output_files.append(output_file)
            logger.info(f"Wrote CDF plot to {output_file}")
    if "violin" in plot_types:
        for variable, values_by_input in values_by_variable.items():
            output_file = output_dir / f"violin_{variable}.png"
            plot_metric_violin_comparison(
                values_by_label=values_by_input,
                variable_name=variable,
                output_file=output_file,
                dpi=dpi,
                colors=colors_by_label,
            )
            output_files.append(output_file)
            logger.info(f"Wrote violin plot to {output_file}")
    if group_by and ("cdf" in plot_types or "violin" in plot_types):
        output_files.extend(
            _write_grouped_metric_plots(
                values_by_variable=values_by_variable,
                output_dir=output_dir,
                output_prefix=output_prefix,
                plot_types=plot_types,
                group_by=group_by,
                label_metadata=label_metadata,
                colors_by_label=colors_by_label,
                linestyles_by_label=linestyles_by_label,
                dpi=dpi,
            )
        )

    if "catchment-map" in plot_types:
        output_files.extend(
            write_metric_catchment_maps(
                input_paths=input_paths,
                variables=variables,
                output_dir=output_dir,
                file_names=file_names,
                shape_paths=shape_paths,
                shape_folder=shape_folder,
                mask_paths=mask_paths,
                mask_folder=mask_folder,
                mask_var=mask_var,
                geometry_match_mode=geometry_match_mode,
                input_names=input_names,
            )
        )
    if _should_write_metric_overview_pdf(
        variables=variables,
        plot_types=plot_types,
        output_files=output_files,
        group_by=group_by,
    ):
        if not values_by_variable:
            try:
                values_by_variable = get_metric_values_by_input(
                    input_paths=input_paths,
                    input_names=input_names,
                    variables=variables,
                    file_names=file_names,
                )
            except ValueError as exc:
                logger.warning(f"Could not create metric summary table: {exc}")
        summary_rows = create_metric_summary_rows(values_by_variable)
        overview_file = write_metric_plot_overview_pdf(
            output_file=output_dir / "metric_plots_overview.pdf",
            plot_files=output_files,
            summary_rows=summary_rows,
            title="Metric plots overview",
            dpi=dpi,
        )
        output_files.append(overview_file)
        logger.info(f"Wrote metric plot overview PDF to {overview_file}")
    return output_files


def _should_write_metric_overview_pdf(
    variables, plot_types, output_files, group_by=None
):
    """Check whether a metric overview PDF should be written.

    Parameters
    ----------
    variables : Sequence[str]
        Selected metric variables.
    plot_types : Sequence[str]
        Selected plot types.
    output_files : Sequence[str or Path]
        Plot files returned by plot writers.
    group_by : Sequence[str], optional
        Grouping fields that create extra plots.

    Returns
    -------
    bool
        True when an overview PDF should be written.
    """
    if group_by:
        return any(Path(output_file).is_file() for output_file in output_files)
    if len(variables) <= 1 and len(plot_types) <= 1:
        return False
    return any(Path(output_file).is_file() for output_file in output_files)


def write_metric_catchment_maps(
    input_paths: Sequence[str],
    variables: Sequence[str],
    output_dir,
    input_names: Optional[Sequence[str]] = None,
    file_names: str = "*.csv",
    shape_paths: Optional[Sequence[str]] = None,
    shape_folder=None,
    mask_paths: Optional[Sequence[str]] = None,
    mask_folder=None,
    mask_var=None,
    geometry_match_mode="auto",
) -> List[Path]:
    """Write catchment median maps from metric CSV inputs.

    Parameters
    ----------
    input_paths : Sequence[str]
        CSV files or directories.
    variables : Sequence[str]
        Metric columns to plot.
    output_dir : str or Path
        Directory for map PNG files.
    input_names : Sequence[str], optional
        Names used for per-realisation map output directories.
    file_names : str, optional
        Glob pattern used for recursive directory searches.
    shape_paths : Sequence[str], optional
        Shapefiles matched one-to-one with input_paths.
    shape_folder : str or Path, optional
        Folder with ID-matched shapefiles.
    mask_paths : Sequence[str], optional
        NetCDF masks matched one-to-one with input_paths.
    mask_folder : str or Path, optional
        Folder with ID-matched NetCDF masks.
    mask_var : str, optional
        Mask variable name.
    geometry_match_mode : str, optional
        Geometry matching mode. Only "auto" is currently supported.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    if geometry_match_mode != "auto":
        msg = f"Unsupported geometry_match_mode {geometry_match_mode!r}"
        raise ValueError(msg)
    shape_paths = _none_if_empty_sequence(shape_paths)
    mask_paths = _none_if_empty_sequence(mask_paths)
    if shape_paths is not None and mask_paths is not None:
        msg = "Use either shape_paths or mask_paths, not both."
        raise ValueError(msg)
    names = get_metric_input_names(input_paths, input_names)
    split_by_input = len(input_paths) > 1
    if shape_paths is not None:
        if len(shape_paths) != len(input_paths):
            msg = (
                f"Received {len(shape_paths)} geometry paths for "
                f"{len(input_paths)} input paths."
            )
            raise ValueError(msg)
        output_files = []
        for input_path, input_name, shape_path in zip(input_paths, names, shape_paths):
            output_files.extend(
                _write_metric_catchment_maps_for_explicit_geometry(
                    input_paths=[input_path],
                    geometry_paths=[shape_path],
                    variables=variables,
                    output_dir=_get_realisation_output_dir(
                        output_dir, input_name, split_by_input
                    ),
                    file_names=file_names,
                    geometry_kind="shape",
                    title_context=input_name if split_by_input else None,
                )
            )
        return output_files
    if mask_paths is not None:
        if len(mask_paths) != len(input_paths):
            msg = (
                f"Received {len(mask_paths)} geometry paths for "
                f"{len(input_paths)} input paths."
            )
            raise ValueError(msg)
        output_files = []
        for input_path, input_name, mask_path in zip(input_paths, names, mask_paths):
            output_files.extend(
                _write_metric_catchment_maps_for_explicit_geometry(
                    input_paths=[input_path],
                    geometry_paths=[mask_path],
                    variables=variables,
                    output_dir=_get_realisation_output_dir(
                        output_dir, input_name, split_by_input
                    ),
                    file_names=file_names,
                    geometry_kind="mask",
                    mask_var=mask_var,
                    title_context=input_name if split_by_input else None,
                )
            )
        return output_files

    output_files = []
    for input_path, input_name in zip(input_paths, names):
        metric_rows = []
        csv_files = get_metric_csv_files(input_path, file_names=file_names)
        for csv_file in csv_files:
            metric_df = pd.read_csv(csv_file)
            if "id" in metric_df.columns:
                metric_rows.append(
                    calculate_metric_medians(
                        metric_df=metric_df,
                        variables=variables,
                        id_col="id",
                    )
                )
                continue
            match_id = _get_csv_geometry_match_id(
                csv_file=csv_file,
                input_path=input_path,
                shape_folder=shape_folder,
                mask_folder=mask_folder,
            )
            metric_rows.append(
                _calculate_single_metric_median(
                    metric_df=metric_df,
                    variables=variables,
                    row_id=match_id,
                )
            )
        if not metric_rows:
            logger.warning(f"No metric CSV rows available for {input_name}")
            continue
        metric_df = pd.concat(metric_rows, ignore_index=True)
        output_files.extend(
            write_catchment_median_maps(
                metric_df=metric_df,
                output_dir=_get_realisation_output_dir(
                    output_dir, input_name, split_by_input
                ),
                variables=variables,
                shape_folder=shape_folder,
                mask_folder=mask_folder,
                mask_var=mask_var,
                title_context=input_name if split_by_input else None,
            )
        )
    return output_files


def _write_metric_catchment_maps_for_explicit_geometry(
    input_paths,
    geometry_paths,
    variables,
    output_dir,
    file_names="*.csv",
    geometry_kind="shape",
    mask_var=None,
    title_context=None,
):
    """Write catchment maps for explicit one-to-one geometry paths.

    Parameters
    ----------
    input_paths : Sequence[str]
        CSV files or directories.
    geometry_paths : Sequence[str]
        Geometry files matched one-to-one with input_paths.
    variables : Sequence[str]
        Metric columns to plot.
    output_dir : str or Path
        Directory for map PNG files.
    file_names : str, optional
        Glob pattern used for recursive directory searches.
    geometry_kind : str, optional
        Either "shape" or "mask".
    mask_var : str, optional
        Mask variable name.
    title_context : str, optional
        Context added to map titles.

    Returns
    -------
    list[Path]
        Written PNG files.
    """
    if len(geometry_paths) != len(input_paths):
        msg = (
            f"Received {len(geometry_paths)} geometry paths for "
            f"{len(input_paths)} input paths."
        )
        raise ValueError(msg)

    metric_rows = []
    shape_files_by_id = {}
    mask_files_by_id = {}
    for input_path, geometry_path in zip(input_paths, geometry_paths):
        csv_files = get_metric_csv_files(input_path, file_names=file_names)
        metric_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])
        row_id = Path(input_path).stem or Path(input_path).name
        row_id = str(row_id)
        metric_rows.append(
            _calculate_single_metric_median(
                metric_df=metric_df,
                variables=variables,
                row_id=row_id,
            )
        )
        if geometry_kind == "shape":
            shape_files_by_id[row_id] = Path(geometry_path)
        else:
            mask_files_by_id[row_id] = Path(geometry_path)

    metric_df = pd.concat(metric_rows, ignore_index=True)
    return write_catchment_median_maps(
        metric_df=metric_df,
        output_dir=output_dir,
        variables=variables,
        shape_files_by_id=shape_files_by_id or None,
        mask_files_by_id=mask_files_by_id or None,
        mask_var=mask_var,
        title_context=title_context,
    )


def _calculate_single_metric_median(metric_df, variables, row_id):
    """Calculate one median metric row.

    Parameters
    ----------
    metric_df : pandas.DataFrame
        Metric rows.
    variables : Sequence[str]
        Metric columns to aggregate.
    row_id : str
        ID assigned to the output row.

    Returns
    -------
    pandas.DataFrame
        One-row median metric table.
    """
    row = {"id": row_id}
    for variable in variables:
        if variable not in metric_df.columns:
            logger.warning(
                f"Column {variable!r} missing for catchment map row {row_id}"
            )
            row[variable] = np.nan
            continue
        values = pd.to_numeric(metric_df[variable], errors="coerce")
        row[variable] = values.replace([np.inf, -np.inf], np.nan).median()
    return pd.DataFrame([row])


def _get_csv_geometry_match_id(
    csv_file, input_path, shape_folder=None, mask_folder=None
):
    """Get the best geometry match ID for one CSV file.

    Parameters
    ----------
    csv_file : str or Path
        Metric CSV file.
    input_path : str or Path
        Original input path.
    shape_folder : str or Path, optional
        Folder with shapefiles.
    mask_folder : str or Path, optional
        Folder with mask files.

    Returns
    -------
    str
        ID used for geometry matching.
    """
    csv_file = Path(csv_file)
    input_path = Path(input_path)
    candidates = [
        csv_file.stem,
        csv_file.parent.name,
        input_path.stem,
        input_path.name,
    ]
    for candidate in candidates:
        if find_matching_geometry_file(shape_folder, candidate, ".shp") is not None:
            return candidate
        if find_matching_geometry_file(mask_folder, candidate, ".nc") is not None:
            return candidate
    return candidates[0]
