"""Allow ``python -m mhm_tools`` execution."""

from ._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
