# -*- coding: utf-8 -*-

""" Federated algorithm

This file contains all algorithm pieces that are executed on the nodes.
It is important to note that the master method is also triggered on a
node just the same as any other method.

When a return statement is reached the result is sent to the central server.
"""
import time
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.cluster import KMeans
from vantage6.tools.util import info


def master(
        client, data: pd.DataFrame, k: int, columns: list = None,
        org_ids: list = None
) -> dict:
    """ Master algorithm

    Parameters
    ----------
    client
        Vantage6 user or mock client
    data
        DataFrame with the input data
    k
        Number of clusters to be computed
    columns
        Columns to be used for clustering
    org_ids
        List with organisation ids to be used

    Returns
    -------
    results
        Dictionary with the final averaged result
    """

    # Get all organization ids that are within the collaboration,
    # if they were not provided
    # FlaskIO knows the collaboration to which the container belongs
    # as this is encoded in the JWT (Bearer token)
    info('Collecting participating organizations')
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get('id') for organization in organizations
           if not org_ids or organization.get('id') in org_ids]

    # Initialise K global cluster centres, for now start with k random points
    # from a random data node
    info('Initializing k global cluster centres')
    input_ = {
        'method': 'initialize_centroids_partial',
        'kwargs': {'k': k, 'columns': columns}
    }
    task = client.create_new_task(
        input_=input_,
        organization_ids=[np.random.choice(ids)]
    )
    task_id = task.get('id')
    task = client.get_task(task_id)
    while not task.get('complete'):
        task = client.get_task(task_id)
        info('Waiting for results')
        time.sleep(1)
    results = client.get_results(task_id=task.get('id'))
    centroids = results[0]['initial_centroids']

    # Loop until convergence
    iteration = 1
    change = 1.0
    epsilon = 0.05

    while change > epsilon:
        # The input for the partial algorithm
        info('Defining input parameters')
        input_ = {
            'method': 'kmeans_partial',
            'kwargs': {'k': k, 'centroids': centroids, 'columns': columns}
        }

        # Create a new task for the desired organizations
        info('Dispatching node-tasks')
        task = client.create_new_task(
            input_=input_,
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

        # Average centroids by running kmeans on local results
        new_centroids = []
        for result in results:
            for centre in result['centroids']:
                new_centroids.append(centre)
        X = np.array(new_centroids)
        kmeans = KMeans(
            n_clusters=k, random_state=0, n_init='auto',
        ).fit(X)
        new_centroids = kmeans.cluster_centers_

        # Check convergence
        change = 0
        for i in range(k):
            diff = new_centroids[i] - np.array(centroids[i])
            change += np.linalg.norm(diff)
        info(f'Change: {change}, Iteration: {iteration}')

        centroids = list(list(centre) for centre in new_centroids)
        iteration += 1
        if iteration == 300:
            break

    # Final result
    info('Master algorithm complete')
    info(f'Result: {centroids}')

    return {
        'centroids': centroids
    }


def RPC_initialize_centroids_partial(
        data: pd.DataFrame, k: int, columns: list = None
) -> dict:
    # TODO: use a better method to initialize centroids
    info(f'Randomly sample {k} data points to use as initial centroids')
    if columns:
        df = data[columns].sample(k)
    else:
        df = data.sample(k)
    centroids = []
    for index, row in df.iterrows():
        centroids.append(list(row.values))
    return {'initial_centroids': centroids}


def RPC_kmeans_partial(
        data: pd.DataFrame, k: int, centroids: list, columns: list = None
) -> dict:
    """ Partial method for federated kmeans

    Parameters
    ----------
    data
        DataFrame with input data
    k
        Number of clusters to be computed
    centroids
        Initial cluster centroids
    columns
        List with columns to be used for kmeans, if none is given use everything

    Returns
    -------
    results
        Dictionary with the partial result
    """
    info('Selecting columns')
    if columns:
        df = data[columns]

    info('Calculating distance matrix')
    distances = np.zeros([len(df), k])
    for i in range(len(df)):
        for j in range(k):
            xi = list(df.iloc[i].values)
            xj = centroids[j]
            distances[i, j] = distance.euclidean(xi, xj)

    info('Calculating local membership matrix')
    membership = np.zeros([len(df), k])
    for i in range(len(df)):
        j = np.argmin(distances[i])
        membership[i, j] = 1

    info('Generating local cluster centres')
    centroids = []
    for i in range(k):
        members = membership[:, i]
        dfc = df.iloc[members == 1]
        center = []
        for column in columns:
            center.append(dfc[column].mean())
        centroids.append(center)

    return {
        'centroids': centroids
    }
