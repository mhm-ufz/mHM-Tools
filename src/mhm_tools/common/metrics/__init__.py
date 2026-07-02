"""Metric implementations used by mhm-tools.

This package groups reusable metric routines for comparing model outputs with
reference data. Metric implementations live in dedicated modules such as
``tsm``, ``spaef``, ``esp``, ``waspaef``, and ``mspaef``; ``metrics_handler``
dispatches metric calls and writes CSV outputs.

Files
=====

.. autosummary::
   :toctree:

   ~mhm_tools.common.metrics.esp
   ~mhm_tools.common.metrics.metrics_handler
   ~mhm_tools.common.metrics.mspaef
   ~mhm_tools.common.metrics.spaef
   ~mhm_tools.common.metrics.tsm
   ~mhm_tools.common.metrics.waspaef
"""
