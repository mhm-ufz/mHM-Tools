"""Run the heavy parts of a tool on a local dask cluster."""

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def resolve_ncpus(ncpus=None):
    """Turn a requested core count into the number of cores to use.

    Args:
        ncpus: Cores to use, 0 or None for every core the machine reports.

    Returns
    -------
        The core count as a positive int.
    """
    available = os.cpu_count() or 1
    if ncpus is None or int(ncpus) <= 0:
        return available
    requested = int(ncpus)
    if requested > available:
        logger.warning(
            f"{requested} cores were asked for but the machine reports "
            f"{available}, so {available} are used."
        )
        return available
    return requested


@contextmanager
def create_dask_cluster(ncpus=None, use_processes=False):
    """Run a block on a local dask cluster of the given size.

    A chunked record is computed by dask, which without a cluster uses its own
    defaults. Sizing the cluster makes the core count a setting of the tool
    rather than a property of the machine it happens to run on. One core skips
    the cluster, because starting one would only add overhead.

    Args:
        ncpus: Cores to use, 0 or None for every core the machine reports.
        use_processes: Run the workers as processes instead of threads. Threads
            share the arrays, while processes copy them between workers.

    Returns
    -------
        The dask ``Client``, or None when the block runs on one core.
    """
    cores = resolve_ncpus(ncpus)
    if cores == 1:
        logger.info("Running on one core, so no dask cluster is started.")
        yield None
        return

    from dask.distributed import Client, LocalCluster

    # distributed narrates its whole startup and shutdown at info level, which
    # would bury the messages of the tool that started it
    for noisy in ("distributed", "bokeh", "tornado"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cluster = LocalCluster(
        processes=use_processes,
        n_workers=cores if use_processes else 1,
        threads_per_worker=1 if use_processes else cores,
        dashboard_address=None,
    )
    client = Client(cluster)
    logger.info(
        f"Started a local dask cluster on {cores} core(s) as "
        f"{'processes' if use_processes else 'threads'}."
    )
    try:
        yield client
    finally:
        client.close()
        cluster.close()
        logger.info("Closed the local dask cluster.")
