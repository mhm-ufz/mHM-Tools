---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for mHM-Tools, with links to the rest of the site.
html_theme.sidebar_secondary.remove: true
---

# mHM-Tools

```{image} _static/logo.png
:alt: mHM-Tools Logo
:class: dark-light p-2
:width: 200px
:align: center
:target: https://mhm.pages.ufz.de/mhm-tools
```

mHM-Tools is a toolbox to pre- and post-process data for and from the mesoscale Hydrologic Model (mHM). Everything is available as the command line tool `mhm-tools` and as a Python API.

As of version 0.3 it covers three main use cases:

1. Delineate a catchment and create a local mHM setup from a larger existing setup.
2. Create an mHM setup from raw data.
3. Evaluate your mHM runs quickly and versatilely against river discharge, evapotranspiration, soil moisture, snow or total water storage anomalies.

## Installation

```shell
$ pip install mhm-tools
```

## Documentation and source code

* [Command line reference](cli): every `mhm-tools` command and its options
* [API reference](api): the Python modules behind the commands
* [Release notes](about/release_notes): what is new in each version
* [Online documentation](https://mhm.pages.ufz.de/mhm-tools): the latest version of this site
* [Source code](https://git.ufz.de/mhm/mhm-tools) and [issue tracker](https://git.ufz.de/mhm/mhm-tools/-/issues)
* [About](about/index): changelog, license and authors
* mHM [homepage](https://mhm-ufz.org)

```{toctree}
:hidden:
:maxdepth: 3

cli
api
about/index
```

---

LGPLv3, Copyright © 2026, the mHM-Tools developers from Helmholtz-Zentrum für Umweltforschung GmbH - UFZ. All rights reserved.
