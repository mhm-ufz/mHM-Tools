"""Common ESRI ASCII grid routines."""

import warnings
from pathlib import Path

import numpy as np

from .constants import ESRI_REQ, ESRI_TYPES, NO_DATA


def _is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def _extract_header(file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.genfromtxt(
            file, dtype=str, max_rows=6, usecols=(0, 1), invalid_raise=False
        )


def standardize_header(header):
    """
    Standardize an ASCII grid header dictionary.

    Parameters
    ----------
    header : :class:`dict`
        Raw header as dictionary.

    Returns
    -------
    :class:`dict`
        Standardized header as dictionary.

    Raises
    ------
    ValueError
        If the header is missing required information.
        See :any:`ESRI_REQ`
    """
    header = {n: ESRI_TYPES[n](v) for (n, v) in header.items() if n in ESRI_TYPES}
    # convert cell center to corner information
    if "xllcenter" in header:
        header["xllcorner"] = header["xllcenter"] - 0.5 * header.get("cellsize", 1)
        del header["xllcenter"]
    if "yllcenter" in header:
        header["yllcorner"] = header["yllcenter"] - 0.5 * header.get("cellsize", 1)
        del header["yllcenter"]
    # set nodata value if not present
    header.setdefault("nodata_value", NO_DATA)
    # set default lower-left corner
    header.setdefault("xllcorner", 0.0)
    header.setdefault("yllcorner", 0.0)
    # check required header items
    missing = ESRI_REQ - (set(header) & ESRI_REQ)
    if missing:
        msg = f"standardize_header: missing header information {missing}"
        raise ValueError(msg)
    return header


def read_header(file):
    """
    Read an ASCII grid header from file.

    Parameters
    ----------
    file : :class:`~os.PathLike`
        File containing the ASCII grid header.

    Returns
    -------
    :class:`dict`
        Standardized header as dictionary.

    Notes
    -----
    "xllcenter" and "yllcenter" will be converted to
    "xllcorner" and "yllcorner" resepectively.
    """
    header_lines = _extract_header(file)
    return standardize_header(dict(header_lines))


def read_grid(file, dtype=None):
    """
    Read an ASCII grid from file.

    Parameters
    ----------
    file : :class:`~os.PathLike`
        File containing the ASCII grid.
    dtype : str/type, optional
        Data type.
        Needs to be integer or float and compatible with np.dtype
        (i.e. "i4", "f4", "f8"), by default None

    Returns
    -------
    header : dict
        Header describing the grid.
    data : numpy.ndarray
        Data of the grid.

    Raises
    ------
    ValueError
        If data shape is not matching the given header.
    """
    header_lines = _extract_header(file)
    header = standardize_header(dict(header_lines))
    # last line could already be data if "nodata_value" is missing
    numeric_last = _is_number(header_lines[-1][0])
    header_size = len(header_lines) - int(numeric_last)
    data = np.loadtxt(file, dtype=dtype, skiprows=header_size, ndmin=2)
    nrows, ncols = header["nrows"], header["ncols"]
    if data.shape[0] != nrows or data.shape[1] != ncols:
        msg = (
            f"read_grid: data shape {data.shape} "
            f"not matching given header ({nrows=}, {ncols=})."
        )
        raise ValueError(msg)
    return header, data


def write_header(file, header, dtype="f4"):
    """
    Write an ascii header to file.

    Parameters
    ----------
    file : PathLike
        Path the the output file.
    header : dict
        Header describing the grid.
    dtype : str, optional
        Data type.
        Needs to be integer or float and compatible with np.dtype
        (i.e. "i4", "f4", "f8"), by default "f4"
    """
    write_grid(file, header, dtype=dtype)


def write_grid(file, header, data=None, dtype="f4"):
    """
    Write an ascii grid to file.

    Parameters
    ----------
    file : PathLike
        Path the the output file.
    header : dict
        Header describing the grid.
    data : arraylike, optional
        Data of the grid. If not given, only header will be written,
        by default None
    dtype : str, optional
        Data type.
        Needs to be integer or float and compatible with np.dtype
        (i.e. "i4", "f4", "f8"), by default "f4"

    Raises
    ------
    ValueError
        If dtype is neither integer nor float.
    ValueError
        If data is not two dimensional.
    ValueError
        If data shape is not matching the given header.
    """
    header = standardize_header(header)
    if not issubclass(np.dtype(dtype).type, (np.integer, np.floating)):
        msg = f"write_grid: data type needs to be integer or float. Got: {dtype}"
        raise ValueError(msg)
    is_int = issubclass(np.dtype(dtype).type, np.integer)
    if data is not None:
        data = np.array(data, dtype=dtype, copy=False, ndmin=2)
        if data.ndim != 2:
            msg = f"write_grid: data needs to be 2D. Got: {data.ndim}D"
            raise ValueError(msg)
        nrows, ncols = header["nrows"], header["ncols"]
        if data.shape[0] != nrows or data.shape[1] != ncols:
            msg = (
                f"write_grid: data shape {data.shape} "
                f"not matching given header ({nrows=}, {ncols=})."
            )
            raise ValueError(msg)
    # write header and data
    header_path = Path(file)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    with header_path.open("w") as f:
        for key in ["nrows", "ncols", "xllcorner", "yllcorner", "cellsize"]:
            print(key, header[key], file=f)
        typ = int if is_int else float
        print("nodata_value", typ(header["nodata_value"]), file=f)
        if data is not None:
            np.savetxt(f, data, fmt="%i" if is_int else "%f")


def check_resolutions(
    cellsize_1, cellsize_2, first_finer=False, name_1="LA", name_2="LB"
):
    """
    Check two resolutions for compatibility.

    Parameters
    ----------
    cellsize_1 : float
        First cell-size to compare (i.e. finer resolution)
    cellsize_2 : float
        Second cell-size to compare (i.e. coarser resolution)
    first_finer : bool, optional
        Whether to force the first given cell-size to be finer, by default False
    name_1 : str, optional
        Name of the first grid/level, by default "LA"
    name_2 : str, optional
        Name of the second grid/level, by default "LB"

    Returns
    -------
    ratio : int
        Cell-size ratio of coarse/fine.

    Raises
    ------
    ValueError
        If a should be finer than b but isn't.
    ValueError
        If cell factor is not an integer.
    """
    if first_finer and cellsize_1 > cellsize_2:
        msg = (
            "Cell Size missmatch: "
            f"{name_1} ({cellsize_1}) should be finer than "
            f"{name_2} ({cellsize_2})"
        )
        raise ValueError(msg)
    f_ratio = (
        cellsize_1 / cellsize_2 if cellsize_1 > cellsize_2 else cellsize_2 / cellsize_1
    )
    ratio = np.rint(f_ratio).astype(int)
    # same check as done by mHM
    if not np.isclose(ratio, f_ratio, atol=1e-7, rtol=0.0):
        msg = (
            "Cell Size missmatch: "
            f"{name_1} ({cellsize_1}) and "
            f"{name_2} ({cellsize_2}) are not compatible. "
            f"Ratio: {f_ratio}"
        )
        raise ValueError(msg)
    return ratio


def _get_extends(in_size, out_size, nrows, ncols, in_name, out_name):
    """Get extends of new grid."""
    ratio = check_resolutions(
        cellsize_1=in_size,
        cellsize_2=out_size,
        first_finer=True,
        name_1=in_name,
        name_2=out_name,
    )
    # make sure the coarser grid overlaps the finer grid
    ncols_out = ncols // ratio + int(ncols % ratio > 0)
    nrows_out = nrows // ratio + int(nrows % ratio > 0)
    return ncols_out, nrows_out


def rescale_grid(header, cellsize, in_name="LA", out_name="LB"):
    """
    Rescale grid from given header to a coarser cell-size with matching extend.

    Parameters
    ----------
    header : :class:`dict`
        ASCII grid header as dictionary.
    cellsize : :class:`float`
        Target cell-size. Needs to be coarser than the input grid.
    in_name : str, optional
        Name of input grid for error messages, by default "LA"
    out_name : str, optional
        Name of output grid for error messages, by default "LB"

    Returns
    -------
    header : :class:`dict`
        New ASCII grid header as dictionary.

    Raises
    ------
    ValueError
        If given cell-size is not compatible with given grid header.
    """
    result = standardize_header(header)
    ncols, nrows = _get_extends(
        result["cellsize"],
        cellsize,
        result["nrows"],
        result["ncols"],
        in_name,
        out_name,
    )
    result["cellsize"] = float(cellsize)
    result["ncols"] = ncols
    result["nrows"] = nrows
    return result


def check_grid_compatibility(header_1, header_2, name_1="LA", name_2="LB"):
    """
    Check grids for compatibility.

    Parameters
    ----------
    header_1 : dict
        Header of the first gird to check.
    header_2 : dict
        Header of the second grid to check.
    name_1 : str, optional
        Name of the first grid/level, by default "LA"
    name_2 : str, optional
        Name of the second grid/level, by default "LB"

    Raises
    ------
    ValueError
        If the grids don't share the same lower-left corner.
    ValueError
        If the extends are not fitting to be used in mHM.
    """
    header_1 = standardize_header(header_1)
    header_2 = standardize_header(header_2)
    if (
        header_1["xllcorner"] != header_2["xllcorner"]
        or header_1["yllcorner"] != header_2["yllcorner"]
    ):
        msg = (
            "Lower-left corner missmatch: "
            f"{name_1} ({header_1['xllcorner']}, {header_1['yllcorner']})  and "
            f"{name_2} ({header_2['xllcorner']}, {header_2['yllcorner']})  and "
            "don't share the same lower-left corner."
        )
        raise ValueError(msg)
    # find the finer grid
    if header_1["cellsize"] > header_2["cellsize"]:
        header_1, header_2, name_1, name_2 = header_2, header_1, name_2, name_1

    ncols, nrows = _get_extends(
        in_size=header_1["cellsize"],
        out_size=header_2["cellsize"],
        nrows=header_1["nrows"],
        ncols=header_1["ncols"],
        in_name=name_1,
        out_name=name_2,
    )
    if ncols != header_2["ncols"] or nrows != header_2["nrows"]:
        msg = (
            "Extend missmatch: "
            f"{name_2} (ncols={header_2['ncols']}, nrows={header_2['nrows']}) "
            f"would need an extend of ({ncols=}, {nrows=}) "
            f"to be compatible with {name_1}."
        )
        raise ValueError(msg)
