# -*- coding: utf-8 -*-

""" Helper functions for running federated kmeans
"""
import time
from vantage6.tools.util import info


def coordinate_task(client, input: dict, ids: list) -> list:
    """ Coordinate tasks to be sent to data nodes, which includes dispatching
    the task, waiting for results to return and collect completed results

    Parameters
    ----------
    client
        Vantage6 user or mock client
    input
        Input parameters for the task, such as the method and its arguments
    ids
        List with organisation ids that will receive the task

    Returns
    -------
    results
        Collected partial results from all the nodes
    """

    # Create a new task for the desired organizations
    info('Dispatching node tasks')
    task = client.create_new_task(
        input_=input,
        organization_ids=ids
    )

    # Wait for nodes to return results
    info('Waiting for results')
    task_id = task.get('id')
    task = client.get_task(task_id)
    while not task.get('complete'):
        task = client.get_task(task_id)
        info('Waiting for results')
        time.sleep(1)

    # Collecting results
    info('Obtaining results')
    results = client.get_results(task_id=task.get('id'))

    return results
