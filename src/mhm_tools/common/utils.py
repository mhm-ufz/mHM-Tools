"""Utility helpers."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)





def dict_to_multiline_string(d: dict, spacing: int = 12) -> str:
    r"""
    Convert a dictionary into a formatted multiline string.

    Example:
        >>> dict_to_multiline_string({'a': 'b', 'c': 'd'})
        'a           b\nc           d'
    """
    lines = []
    for k, v in d.items():
        lines.append(f"{k!s:<{spacing}}{v}")
    return "\n".join(lines)


def pretty_print_df(df: pd.DataFrame, max_col_width: int = 30) -> None:
    """Pretty-print a DataFrame as an ASCII table with simple truncation.

    Numbers are right-aligned, other columns are left-aligned. Cells longer than
    max_col_width are truncated with an ellipsis.
    """
    if df.empty:
        logger.info("There are no results to display.")
        return

    def is_numeric(col: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(col)

    def fmt_cell(val: object, width: int, right: bool) -> str:
        s = ""
        if not pd.isna(val):
            try:
                val = float(val)
                if val < 10:
                    s = f"{val:.1f}"
                elif val < 1:
                    s = f"{val:.2f}"
                elif val < 0.1:
                    s = f"{val:.3f}"
                elif val < 0.01:
                    s = f"{val:.4f}"
                else:
                    s = f"{val:.0f}"
            except ValueError:
                s = str(val)
        else:
            s = "NaN"
        if len(s) > width:
            s = s[: max(1, width - 1)] + "…"
        return s.rjust(width) if right else s.ljust(width)

    headers = list(df.columns)
    widths = []
    aligns_right = []
    for h in headers:
        col = df[h]
        right = is_numeric(col)
        aligns_right.append(right)
        max_len = max(len(str(h)), *(len(str(x)) for x in col.fillna("")))
        widths.append(min(max_col_width, max_len))

    def sep() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    # Header
    out_string = "\n"
    out_string += sep() + "\n"
    header_cells = [" " + fmt_cell(h, w, False) + " " for h, w in zip(headers, widths)]
    out_string += "|" + "|".join(header_cells) + "|\n"
    out_string += sep() + "\n"

    # Rows
    for _, row in df.iterrows():
        cells = []
        for h, w, right in zip(headers, widths, aligns_right):
            cells.append(" " + fmt_cell(row[h], w, right) + " ")
        out_string += "|" + "|".join(cells) + "|\n"
    out_string += sep() + "\n"

    logger.info(out_string)
