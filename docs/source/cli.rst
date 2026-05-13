=========
CLI usage
=========

The command line interface uses a single base command ``mhm-tools``.
All features are then provided as sub-commands as documented here.

.. note::
   You can get help for each (sub-)command with the option ``-h`` or ``--help``.

The CLI is based on Click and supports shell autocompletion.

Autocompletion
--------------

For the current shell session:

- ``bash``:
  ``eval "$(_MHM_TOOLS_COMPLETE=bash_source mhm-tools)"``
- ``zsh``:
  ``eval "$(_MHM_TOOLS_COMPLETE=zsh_source mhm-tools)"``
- ``fish``:
  ``eval (env _MHM_TOOLS_COMPLETE=fish_source mhm-tools)``

To persist completion across sessions, add the corresponding command to your
shell startup file (for example ``~/.bashrc`` or ``~/.zshrc``).

Command suggestions
-------------------

If a command name is misspelled, the CLI suggests close matches, e.g.:

``No such command 'run_ovverview'. Did you mean: run_overview?``
