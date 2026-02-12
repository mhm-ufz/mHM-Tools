"""Allow ``python -m mhm_tools._cli`` execution."""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
