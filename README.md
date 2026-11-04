# mHM Tools

<a href="https://mhm-tools.readthedocs.io" title="mHM-Tools documentation" target="_blank">
  <img width="300" src="https://git.ufz.de/mhm/mhm-tools/-/raw/main/docs/source/_static/logo_large.png" />
</a>

mHM-Tools is a toolbox to pre- and post-process data for and from the mesoscale Hydrologic Model (mHM). Everything is available as the command line tool `mhm-tools` and as a Python API.

As of version 0.3 it covers three main use cases:

1. Delineate a catchment and create a local mHM setup from a larger existing setup.
2. Create an mHM setup from raw data.
3. Evaluate your mHM runs against river discharge, evapotranspiration, soil moisture, snow or total water storage anomalies.

## Installation

```shell
$ pip install mhm-tools
```

To install from a source checkout into a clean conda environment, which nothing else in your shell interferes with (e.g. on an HPC system or to test a release), use the install script. It creates the environment if needed, ignores `PYTHONPATH`, the user site-packages and pip settings, and checks the result:
```shell
$ scripts/install_mhm_tools.sh /path/to/envs/mhm-tools [--editable] [--extras test,doc] [--python 3.12]
$ conda activate /path/to/envs/mhm-tools
```

For a first overview how to run the model you can use the help function of the CLI: 
```shell
$ mhm-tools -h
```
or the interactive terminal user interface (TUI)
```shell
$ mhm-tools tui
```

## Documentation and source code

* [Documentation](https://mhm-tools.readthedocs.io), with the [command line reference](https://mhm-tools.readthedocs.io/en/latest/cli.html) and the [API reference](https://mhm-tools.readthedocs.io/en/latest/api.html)

  We are currently migrating the documentation to Read the Docs. The previous version is still available on [GitLab Pages](https://mhm.pages.ufz.de/mhm-tools).
* [Source code](https://git.ufz.de/mhm/mhm-tools) and [issue tracker](https://git.ufz.de/mhm/mhm-tools/-/issues)
* [Release notes](https://git.ufz.de/mhm/mhm-tools/-/blob/main/RELEASE_NOTES.md) and the full [changelog](https://git.ufz.de/mhm/mhm-tools/-/blob/main/CHANGELOG.md)
* mHM [homepage](https://mhm-ufz.org)

## License

LGPLv3, Copyright © 2026, the mHM-Tools developers from Helmholtz-Zentrum für Umweltforschung GmbH - UFZ. All rights reserved.
