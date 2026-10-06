# mHM Tools

<a href="https://mhm.pages.ufz.de/mhm-tools" title="mHM-Tools documentation" target="_blank">
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

For a first overview how to run the model you can use the help function of the CLI: 
```shell
$ mhm-tools -h
```
or the interactive terminal user interface (TUI)
```shell
$ mhm-tools tui
```

## Documentation and source code

* [Documentation](https://mhm.pages.ufz.de/mhm-tools), with the [command line reference](https://mhm.pages.ufz.de/mhm-tools/cli.html) and the [API reference](https://mhm.pages.ufz.de/mhm-tools/api.html)
* [Source code](https://git.ufz.de/mhm/mhm-tools) and [issue tracker](https://git.ufz.de/mhm/mhm-tools/-/issues)
* [Release notes](https://git.ufz.de/mhm/mhm-tools/-/blob/main/RELEASE_NOTES.md) and the full [changelog](https://git.ufz.de/mhm/mhm-tools/-/blob/main/CHANGELOG.md)
* mHM [homepage](https://mhm-ufz.org)

## License

LGPLv3, Copyright © 2026, the mHM-Tools developers from Helmholtz-Zentrum für Umweltforschung GmbH - UFZ. All rights reserved.
