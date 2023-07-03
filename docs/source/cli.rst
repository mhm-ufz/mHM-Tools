=========
CLI usage
=========

The command line interface uses a single base command ``mhm-tools``.
All features are then provided as sub-commands as documented here.

.. note::
   You can get help for each (sub-)command with the option ``-h`` or ``--help``.

.. sphinx_argparse_cli::
  :module: mhm_tools._cli._main
  :func: _get_parser
  :prog: mhm-tools
