"""Read and convert the CF style units of hydrological amounts and rates.

An amount is a volume (m3, l, km3, ...) or a depth of water (mm, m, kg m-2,
where 1 kg/m2 of water is 1 mm). A rate is an amount per time unit, such as
m3 s-1, mm/d or kg m-2 s-1. Temperatures are converted separately, because
their units differ by an offset rather than a factor.

Authors
-------
- Simon Lüdke
"""

import re
from typing import NamedTuple, Optional

import numpy as np

from mhm_tools.common.constants import AMOUNT_UNITS, TIME_UNIT_SECONDS

# characters and phrases rewritten before a unit is split into its parts
UNIT_TEXT_REPLACEMENTS = (
    ("³", "3"),
    ("²", "2"),
    ("¹", "1"),
    ("⁻", "-"),
    ("\u2212", "-"),  # minus sign
    ("^", ""),
    ("**", ""),
    ("·", " "),
    ("*", " "),
    (" of water", ""),
    (" water equivalent", ""),
    (" h2o", ""),
)
UNIT_NAME_ALIASES = {
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "day": "d",
    "days": "d",
    "mon": "month",
    "months": "month",
    "yr": "year",
    "yrs": "year",
    "years": "year",
    "litre": "l",
    "liter": "l",
    "litres": "l",
    "liters": "l",
    "millimetre": "mm",
    "millimeter": "mm",
    "millimetres": "mm",
    "millimeters": "mm",
    "centimetre": "cm",
    "centimeter": "cm",
    "centimetres": "cm",
    "centimeters": "cm",
    "metre": "m",
    "meter": "m",
    "metres": "m",
    "meters": "m",
}
CELSIUS_UNITS = {
    "c",
    "°c",
    "degc",
    "celsius",
    "deg_c",
    "degree_c",
    "degrees_c",
    "degree_celsius",
    "degrees_celsius",
}
KELVIN_UNITS = {"k", "kelvin"}
FAHRENHEIT_UNITS = {
    "f",
    "°f",
    "degf",
    "fahrenheit",
    "deg_f",
    "degree_f",
    "degrees_f",
    "degree_fahrenheit",
    "degrees_fahrenheit",
}


class UnitParts(NamedTuple):
    """Parts of an amount or rate unit.

    Attributes
    ----------
    kind : str
        "volume" or "depth".
    amount_factor : float
        Factor converting the amount to m3 (volume) or metres of water (depth).
    time_seconds : float or None
        Length of the time unit of a rate in seconds, None for an amount.
    amount_units : str
        Canonical text of the amount, e.g. "m3" or "kg m-2".
    """

    kind: str
    amount_factor: float
    time_seconds: Optional[float]
    amount_units: str


def _normalize_unit_text(units):
    """Lower case a unit and rewrite its exponent and separator variants.

    Args:
        units: Unit string.

    Returns
    -------
        The rewritten unit string.
    """
    text = str(units).strip().lower()
    for old, new in UNIT_TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def split_units(units):
    """Split an amount or rate unit such as "m3 s-1", "mm/d" or "kg m-2".

    Args:
        units: CF style unit string.

    Returns
    -------
        The `UnitParts` of the unit. Raises ``ValueError`` for a unit that is no
        known amount or amount per time unit.
    """
    numerator, *denominators = _normalize_unit_text(units).split("/")
    exponents = {}
    for part, sign in [(numerator, 1), *((part, -1) for part in denominators)]:
        for token in part.replace(".", " ").split():
            match = re.fullmatch(r"([a-z]+)(-?\d+)?", token)
            if match is None:
                msg = f"Cannot read the units {units!r}."
                raise ValueError(msg)
            name = UNIT_NAME_ALIASES.get(match.group(1), match.group(1))
            power = int(match.group(2) or 1)
            exponents[name] = exponents.get(name, 0) + sign * power
    time_units = [name for name in exponents if name in TIME_UNIT_SECONDS]
    if len(time_units) > 1 or any(exponents[name] != -1 for name in time_units):
        msg = f"{units!r} is no amount per single time unit."
        raise ValueError(msg)
    time_seconds = TIME_UNIT_SECONDS[time_units[0]] if time_units else None
    amount = {
        name: power for name, power in exponents.items() if name not in time_units
    }
    amount_key = tuple(sorted((name, power) for name, power in amount.items() if power))
    if amount_key not in AMOUNT_UNITS:
        msg = f"Unknown units {units!r}, expected a volume or depth of water."
        raise ValueError(msg)
    kind, amount_factor = AMOUNT_UNITS[amount_key]
    amount_units = " ".join(
        name if power == 1 else f"{name}{power}"
        for name, power in amount.items()
        if power
    )
    return UnitParts(kind, amount_factor, time_seconds, amount_units)


def create_units_string(amount_units, time_unit=None):
    """Create a canonical CF unit string from an amount and a time unit.

    Args:
        amount_units: Canonical amount, e.g. "mm" or "m3".
        time_unit: Name of the time unit, e.g. "d", None for an amount.

    Returns
    -------
        Unit string such as "mm d-1" or "m3".
    """
    if time_unit is None:
        return amount_units
    return f"{amount_units} {UNIT_NAME_ALIASES.get(time_unit, time_unit)}-1"


def calculate_conversion_factor(from_units, to_units):
    """Calculate the factor converting values between two amounts or two rates.

    Args:
        from_units: Units of the values, e.g. "m3 d-1".
        to_units: Target units, e.g. "m3/s".

    Returns
    -------
        The factor the values are multiplied with. Raises ``ValueError`` when a
        volume meets a depth or an amount meets a rate.
    """
    source = split_units(from_units)
    target = split_units(to_units)
    if source.kind != target.kind:
        msg = (
            f"Cannot convert the {source.kind} {from_units!r} to the "
            f"{target.kind} {to_units!r} without a catchment area."
        )
        raise ValueError(msg)
    if (source.time_seconds is None) != (target.time_seconds is None):
        msg = f"Cannot convert between the amount and rate units {from_units!r} and {to_units!r}."
        raise ValueError(msg)
    factor = source.amount_factor / target.amount_factor
    if source.time_seconds is not None:
        factor *= target.time_seconds / source.time_seconds
    return factor


def calculate_amount_factor(rate_units, step_seconds, to_units, area_km2=None):
    """Calculate the factor turning a rate over one time step into an amount.

    A depth rate becomes a volume, and a volume rate a depth, only with the
    catchment area they apply to.

    Args:
        rate_units: Units of the rate, e.g. "m3 s-1" or "kg m-2 s-1".
        step_seconds: Length of one time step in seconds.
        to_units: Units of the amount, e.g. "m3" or "mm".
        area_km2: Catchment area in km2 for a depth to volume conversion.

    Returns
    -------
        The factor the rate is multiplied with, NaN when a needed area is
        missing. Raises ``ValueError`` when the units are no rate and amount.
    """
    rate = split_units(rate_units)
    target = split_units(to_units)
    if rate.time_seconds is None or target.time_seconds is not None:
        msg = f"Cannot turn {rate_units!r} into an amount in {to_units!r}."
        raise ValueError(msg)
    # amount over one step in m3 or in metres of water
    factor = step_seconds / rate.time_seconds * rate.amount_factor
    if rate.kind == target.kind:
        return factor / target.amount_factor
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return np.nan
    area_m2 = area_km2 * 1e6
    if rate.kind == "depth":
        return factor * area_m2 / target.amount_factor
    return factor / area_m2 / target.amount_factor


def convert_rate_time_unit(units, time_unit):
    """Calculate how a rate changes when it is expressed per another time unit.

    Args:
        units: Units of the rate, e.g. "mm d-1".
        time_unit: Target time unit, e.g. "h" or "hour".

    Returns
    -------
        Tuple of (factor, new units), e.g. (1/24, "mm h-1"). Raises
        ``ValueError`` for an amount or an unknown time unit.
    """
    parts = split_units(units)
    time_name = UNIT_NAME_ALIASES.get(time_unit, time_unit)
    if parts.time_seconds is None or time_name not in TIME_UNIT_SECONDS:
        msg = f"Cannot express {units!r} per {time_unit!r}."
        raise ValueError(msg)
    factor = TIME_UNIT_SECONDS[time_name] / parts.time_seconds
    return factor, create_units_string(parts.amount_units, time_name)


def get_closest_time_unit(seconds):
    """Get the time unit whose length is closest to a time step.

    Args:
        seconds: Length of the time step in seconds.

    Returns
    -------
        Name of the time unit, one of the keys of `TIME_UNIT_SECONDS`.
    """
    # the ratio rather than the difference, so an hour is not taken for a second
    return min(
        TIME_UNIT_SECONDS,
        key=lambda name: abs(np.log(seconds / TIME_UNIT_SECONDS[name])),
    )


def convert_temperature_to_celsius(values, units):
    """Convert temperatures to degrees Celsius.

    Args:
        values: Temperatures as a number, array or DataArray.
        units: Their units, e.g. "K", "degF" or "degree_Celsius".

    Returns
    -------
        The temperatures in degrees Celsius. Raises ``ValueError`` for an
        unknown temperature unit.
    """
    text = _normalize_unit_text(units).replace(" ", "_")
    if text in CELSIUS_UNITS:
        return values
    if text in KELVIN_UNITS:
        return values - 273.15
    if text in FAHRENHEIT_UNITS:
        return (values - 32) * (5 / 9)
    msg = f"Unknown temperature units {units!r}."
    raise ValueError(msg)
