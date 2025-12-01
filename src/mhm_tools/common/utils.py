def dict_to_multiline_string(d: dict, spacing: int = 12) -> str:
    """
    Convert a dictionary into a formatted multiline string.

    Example:
        >>> dict_to_multiline_string({'a': 'b', 'c': 'd'})
        'a           b\nc           d'
    """
    lines = []
    for k, v in d.items():
        lines.append(f"{k!s:<{spacing}}{v}")
    return "\n".join(lines)
