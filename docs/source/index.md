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

### Clean install into a conda environment

To install mhm-tools from a source checkout into a conda environment that nothing else in your shell interferes with, e.g. on an HPC system or to test a release, use the install script:

```shell
$ git clone https://git.ufz.de/mhm/mhm-tools.git
$ cd mhm-tools
$ scripts/install_mhm_tools.sh /path/to/envs/mhm-tools
$ conda activate /path/to/envs/mhm-tools
```

The script needs conda or mamba. It creates the environment if the path does not exist, with the compiled dependencies and the CDO program from conda-forge, or updates an existing one. It ignores `PYTHONPATH`, the user site-packages and the pip settings of your shell, replaces an older mhm-tools install, and checks afterwards that `mhm_tools` is imported from the environment.

* `--editable`: install in editable mode, for development
* `--extras test,doc`: also install optional dependencies, `mhm` or `mhm5` adds mHM from conda-forge, and so does `test`, because the tests need mHM
* `--python 3.12`: Python version of a new environment (default: chosen by conda to fit the dependencies)

### mHM for restart creation

Only `create-mhm-restart-from-setup` needs mHM itself. It runs mHM 5.13 or newer through its Python bindings and checks before it creates any tile that `import mhm` works. Install mHM along with mhm-tools by choosing the major version as an extra:

```shell
$ pip install "mhm-tools[mhm]"    # the newest mHM from 5.13 on
$ pip install "mhm-tools[mhm5]"   # stay on mHM 5 (5.13 or newer)
```

mHM 6 is not available yet, so both currently install mHM 5. Once mHM 6 is released, `mhm` installs it, while `mhm5` keeps mHM 5; an `mhm6` extra will then select mHM 6 explicitly.

With conda, install mHM from conda-forge, or let the install script do it with `--extras mhm` or `--extras mhm5`:

```shell
$ conda install -c conda-forge "mhm>=5.13"       # or "mhm>=5.13,<6" to stay on mHM 5
$ scripts/install_mhm_tools.sh /path/to/envs/mhm-tools --extras mhm5
```

The mHM runs call `python` from `PATH`, so activate the environment that holds mHM before you start the command. On an HPC system, you can use an mHM module instead of installing mHM: `--mhm-packages "/path/to/modulefiles mhm/5.13"` runs `module use` for the paths and `module load` for the modules before each mHM run, which also selects the mHM version.

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
