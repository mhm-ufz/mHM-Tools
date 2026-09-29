"""Module to handle resolution information for MHM tools."""

import logging
from pathlib import Path

import numpy as np

from mhm_tools.common.file_handler import get_coord_values, get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.xarray_utils import get_coord_key

logger = logging.getLogger(__name__)


class Resolution:
    """Class to hold resolution information."""

    def __init__(
        self,
        l1=None,
        l11=None,
        l2=None,
        l2_file=None,
        l0=None,
        l0_resolution=None,
        l1_resolution=None,
        l11_resolution=None,
        l2_resolution=None,
        raise_on_missmatch=True,
    ):
        """Initialize the Resolution class."""
        self.l0 = l0 if l0 is not None else l0_resolution
        self.l1 = l1 if l1 is not None else l1_resolution
        self.l11 = l11 if l11 is not None else l11_resolution
        self.l2 = l2 if l2 is not None else l2_resolution
        self.l0 = float(self.l0) if self.l0 is not None else None
        self.l1 = float(self.l1) if self.l1 is not None else None
        self.l11 = float(self.l11) if self.l11 is not None else None
        self.l2 = float(self.l2) if self.l2 is not None else None
        self.l2_file = l2_file
        if self.l2_file is not None:
            self.l2_file = Path(self.l2_file)
            if self.l2_file.is_dir():
                # get the first netcdf file in the directory
                nc_files = list(self.l2_file.rglob("*.nc"))
                if len(nc_files) == 0:
                    with ErrorLogger(logger):
                        msg = f"No netcdf files found in {self.l2_file}."
                        raise FileNotFoundError(msg)
                self.l2_file = nc_files[0]
            elif not self.l2_file.is_file():
                with ErrorLogger(logger):
                    msg = f"L2 file {self.l2_file} not found."
                    raise FileNotFoundError(msg)
            if self.l2_file.suffix == ".nc":
                with get_xarray_ds_from_file(self.l2_file) as ds:
                    lon = get_coord_values(ds, lon=True)
                    file_res = round(calculate_coordinate_resolution(lon), 9)
                    if self.l2 is not None and abs(file_res - self.l2) > 1e-6:
                        msg = f"Provided l2_resolution {self.l2} differs from resolution derived from file {file_res}. Either provide the correct l2_resolution or remove it to use the resolution derived from the file."
                        if raise_on_missmatch:
                            with ErrorLogger(logger):
                                raise ValueError(msg)
                        logger.warning(msg)
                        self.l2 = file_res
                    elif self.l2 is None:
                        logger.info(
                            f"Derived l2_resolution {file_res} from {self.l2_file}"
                        )
                        self.l2 = file_res
            else:
                logger.error(
                    f"Unsupported file format for l2_file: {self.l2_file.suffix}"
                )
                self.l2_file = None

        self.l11 = self.l11 if self.l11 is not None else self.l1
        self.l2 = self.l2 if self.l2 is not None else self.l1

    @property
    def l0_resolution(self):
        """Backward-compatible alias for l0."""
        return self.l0

    @l0_resolution.setter
    def l0_resolution(self, value):
        self.l0 = value

    @property
    def l1_resolution(self):
        """Backward-compatible alias for l1."""
        return self.l1

    @l1_resolution.setter
    def l1_resolution(self, value):
        self.l1 = value

    @property
    def l11_resolution(self):
        """Backward-compatible alias for l11."""
        return self.l11

    @l11_resolution.setter
    def l11_resolution(self, value):
        self.l11 = value

    @property
    def l2_resolution(self):
        """Backward-compatible alias for l2."""
        return self.l2

    @l2_resolution.setter
    def l2_resolution(self, value):
        self.l2 = value

    def get_max_resolution(self):
        """Get the maximum resolution."""
        return max(
            r
            for r in [
                self.l1,
                self.l11,
                self.l2,
            ]
            if r is not None
        )

    def only_one_resolution(self):
        """Return the resolution if there is only one otherwise None."""
        n = 0
        res = None
        for r in [
            self.l0,
            self.l1,
            self.l11,
            self.l2,
        ]:
            if r is not None and r != res:
                n += 1
                res = r
        if n == 1:
            return res
        return None


def get_file_res(lon=None, lat=None, resolutions=None, ds=None, raise_exception=True):
    """Get resolution from coordinates and match it to provided resolutions if close enough.

    Args:
        lon: Longitude coordinate with at least two values.
        lat: Latitude coordinate, used if lon holds less than two values.
        resolutions (Resolution): Resolutions to snap the derived resolution to.
        ds: Dataset or DataArray to take the coordinates and the
            "spatial_resolution" attribute from instead of lon and lat.
        raise_exception (bool): Raise a ValueError if the resolution cannot be
            determined at all. Returns NaN instead if False. Defaults to True.

    Returns
    -------
        The file resolution as positive float, NaN if it cannot be determined
        and raise_exception is False.
    """
    if resolutions is None:
        resolutions = Resolution()
    attribute_res = None
    if ds is not None:
        lon = ds[get_coord_key(ds, lon=True)]
        lat = ds[get_coord_key(ds, lat=True)]
        attribute_res = ds.attrs.get("spatial_resolution")
    coord = lon if lon is not None and len(lon) > 1 else lat
    if coord is not None and len(coord) > 1:
        file_res = calculate_coordinate_resolution(coord)
    elif attribute_res is not None:
        # nothing to measure against, so the attribute has to be trusted
        file_res = attribute_res
    elif resolutions.only_one_resolution() is not None:
        file_res = resolutions.only_one_resolution()
        logger.warning(
            "Taken resolution from provided resolutions as ds was to small to derive it."
        )
    elif raise_exception:
        with ErrorLogger(logger):
            error_msg = "Cannot determine file resolution: no valid lon or lat coordinates provided (need len > 1)."
            raise ValueError(error_msg)
    else:
        logger.warning(
            "Cannot determine file resolution from the given coordinates, using NaN."
        )
        file_res = float("nan")

    # a resolution is a magnitude; descending coordinates would otherwise
    # yield a negative diff and break the abs()-based matching below
    file_res = abs(file_res)

    # a measured resolution wins over the attribute, which is only reported here
    if attribute_res is not None and abs(file_res - abs(attribute_res)) > 1e-5:
        logger.warning(
            f"Spatial resolution attribute {attribute_res} differs from the resolution "
            f"{file_res} derived from the coordinates. Using the derived resolution."
        )
    elif attribute_res is not None:
        logger.debug(
            f"Spatial resolution attribute {attribute_res} matches the resolution "
            f"{file_res} derived from the coordinates."
        )
        file_res = abs(attribute_res)

    # a resolution is a magnitude; descending coordinates would otherwise
    # yield a negative diff and break the abs()-based matching below
    file_res = abs(file_res)

    # a resolution is a magnitude; descending coordinates would otherwise
    # yield a negative diff and break the abs()-based matching below
    file_res = abs(file_res)

    # if file_res is close to a resolution in resolutions, use that one to avoid floating point issues
    if resolutions.l0 is not None and abs(file_res - resolutions.l0) < 1e-5:
        logger.debug(
            f"File resolution {file_res} is close to l0 resolution {resolutions.l0}. Using l0 resolution."
        )
        return resolutions.l0
    if resolutions.l2 is not None and abs(file_res - resolutions.l2) < 1e-5:
        logger.debug(
            f"File resolution {file_res} is close to l2 resolution {resolutions.l2}. Using l2 resolution."
        )
        return resolutions.l2
    if resolutions.l1 is not None and abs(file_res - resolutions.l1) < 1e-5:
        logger.debug(
            f"File resolution {file_res} is close to l1 resolution {resolutions.l1}. Using l1 resolution."
        )
        return resolutions.l1
    if resolutions.l11 is not None and abs(file_res - resolutions.l11) < 1e-5:
        logger.debug(
            f"File resolution {file_res} is close to l11 resolution {resolutions.l11}. Using l11 resolution."
        )
        return resolutions.l11
    logger.debug(
        f"File resolution {file_res} does not match any provided resolution ({resolutions.l0}, {resolutions.l1}, {resolutions.l2}, {resolutions.l11})."
    )
    return file_res


def calculate_coordinate_resolution(coord, outlier_rtol=0.5):
    """Return the spacing of a 1-D coordinate.

    Uses the average step over the whole coordinate, which averages out the
    rounding noise of the stored values. Falls back to the median step and warns
    if single steps deviate strongly, e.g. for gaps or a wrapped longitude.

    Args:
        coord: Coordinate DataArray or array with at least two values.
        outlier_rtol (float): Relative deviation from the median step above which
            a step counts as a defect. Defaults to 0.5.

    Returns
    -------
        The spacing as positive float.
    """
    values = np.asarray(coord, dtype=float)
    if values.ndim != 1 or values.size < 2:
        msg = (
            f"Cannot determine the resolution of a coordinate with shape "
            f"{values.shape}. Two-dimensional or single-cell grids are not supported."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    steps = np.diff(values)
    median_step = np.nanmedian(steps)
    # few strongly deviating steps are defects (gaps, wrapped longitudes),
    # many slightly deviating ones are only rounding noise of the stored values
    if not np.all(np.isfinite(steps)) or np.any(
        np.abs(steps - median_step) > outlier_rtol * abs(median_step)
    ):
        logger.warning(
            f"Coordinate is not evenly spaced. Using the median step "
            f"{abs(median_step)} instead of the average step."
        )
        return abs(float(median_step))
    return abs(float(values[-1] - values[0]) / (values.size - 1))
