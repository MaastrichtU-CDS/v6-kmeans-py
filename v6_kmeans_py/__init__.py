# -*- coding: utf-8 -*-

""" Federated algorithm for kmeans

We follow the approach introduced by Stallmann and Wilbik (2022),
but implementing it for classical kmeans.
"""
import random

import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from vantage6.tools.util import info
from v6_kmeans_py.helper import coordinate_task


def master(
        client, data: pd.DataFrame, k: int, epsilon: int = 0.05,
        max_iter: int = 300, columns: list = None, org_ids: list = None,
        d_init: str = 'all', init_method: str = 'random',
        avg_method: str = 'k-means'
) -> dict:
    """ Master algorithm that coordinates the tasks and performs averaging

    Parameters
    ----------
    client
        Vantage6 user or mock client
    data
        DataFrame with the input data
    k
        Number of clusters to be computed
    epsilon
        Threshold for convergence criterion
    max_iter
        Maximum number of iterations to perform
    columns
        Columns to be used for clustering
    org_ids
        List with organisation ids to be used
    d_init
        Which data nodes to use for initialisation ('all' or 'random')
    init_method
        Method to be used for centroids initialisation ('random' or 'k-means++')
    avg_method
        Method used to get global centroids ('simple_avg' or 'k-means')

    Returns
    -------
    results
        Dictionary with the final averaged result
    """

    # Get all organization ids that are within the collaboration or
    # use the provided ones
    info('Collecting participating organizations')
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get('id') for organization in organizations
           if not org_ids or organization.get('id') in org_ids]

    # Checking centroids initialisation input parameters
    if d_init not in ['all', 'random']:
        info(f'Initialisation option {d_init} not available, using all nodes')
        d_init = 'all'
    if init_method not in ['random', 'k-means++']:
        info(f'Initialisation option {init_method} not available, using random')
        init_method = 'random'
    if avg_method not in ['simple_avg', 'k-means']:
        info(f'Initialisation option {avg_method} not available, using k-means')
        avg_method = 'k-means'

    # Initialise k global cluster centroids
    info('Initializing k global cluster centres')
    input_ = {
        'method': 'initialize_centroids_partial',
        'kwargs': {'k': k, 'columns': columns, 'method': init_method}
    }
    if d_init == 'all':
        # Draw from all nodes and then re-draw
        results = coordinate_task(client, input_, ids)
        results = [centroid for result in results for centroid in result]
        if init_method == 'random':
            centroids = random.sample(results, k)
        else:
            X = np.array(results)
            centroids, indices = kmeans_plusplus(X, n_clusters=k)
            centroids = centroids.tolist()
    else:
        # Draw from random data node
        random_id = [random.choice(ids)]
        results = coordinate_task(client, input_, random_id)
        centroids = results[0]

    # The next steps are run until convergence is achieved or the maximum
    # number of iterations reached. In order to evaluate convergence,
    # we compute the difference of the centroids between two steps. We
    # initialise the `change` variable to something higher than the threshold
    # epsilon.
    iteration = 1
    change = 2*epsilon
    while (change > epsilon) and (iteration < max_iter):
        # The input for the partial algorithm
        info('Defining input parameters')
        input_ = {
            'method': 'kmeans_partial',
            'kwargs': {'k': k, 'centroids': centroids, 'columns': columns}
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)

        # Get global centroids
        info('Run global averaging for centroids')
        if avg_method == 'k-means':
            info('Running k-Means on local clusters')
            local_centroids = [
                centroid for result in results for centroid in result
            ]
            X = np.array(local_centroids)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            new_centroids = kmeans.cluster_centers_
        else:
            info('Running simple average on local clusters')
            new_centroids = np.zeros([k, len(columns)])
            for i in range(len(ids)):
                for j in range(k):
                    for p in range(len(columns)):
                        new_centroids[j, p] += results[i][j][p]
            new_centroids = new_centroids / len(ids)

        # Compute the sum of the magnitudes of the centroids differences
        # between steps. This change in centroids between steps will be used
        # to evaluate convergence.
        info('Compute change in cluster centroids')
        change = 0
        for i in range(k):
            diff = new_centroids[i] - np.array(centroids[i])
            change += np.linalg.norm(diff)
        info(f'Iteration: {iteration}, change in centroids: {change}')

        # Re-define the centroids and update iterations counter
        centroids = list(list(centre) for centre in new_centroids)
        iteration += 1

    # Final result
    info('Master algorithm complete')
    info(f'Result: {centroids}')

    return {
        'centroids': centroids
    }


def RPC_initialize_centroids_partial(
        data: pd.DataFrame, k: int, columns: list = None, method: str = 'random'
) -> list:
    """ Initialise centroids for kmeans

    Parameters
    ----------
    data
        Dataframe with input data
    k
        Number of clusters
    columns
        Columns to be used for clustering
    method
        Method to be used for centroids initialisation ('random' or 'k-means++')

    Returns
    -------
    centroids
        Initial guess for centroids
    """
    # Drop rows with NaNs
    data = data.dropna(subset=columns)

    # Initialise local centroids
    if method == 'random':
        info(f'Randomly sample {k} data points to use as initial centroids')
        df = data[columns].sample(k)
        centroids = df.values.tolist()
    else:
        info(f'Sampling {k} data points using k-means++')
        X = data[columns].values
        centroids, indices = kmeans_plusplus(X, n_clusters=k)
        centroids = centroids.tolist()

    return centroids


def RPC_kmeans_partial(
        df: pd.DataFrame, k: int, centroids: list, columns: list = None
) -> list:
    """ Partial method for federated kmeans

    Parameters
    ----------
    df
        DataFrame with input data
    k
        Number of clusters to be computed
    centroids
        Initial cluster centroids
    columns
        List with columns to be used for kmeans, if none is given use everything

    Returns
    -------
    centroids
        List with the partial result for centroids
    """
    # Drop rows with NaNs
    df = df.dropna(subset=columns)

    info('Selecting columns')
    if columns:
        df = df[columns]

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

    info('Generating local cluster centroids')
    centroids = []
    for i in range(k):
        members = membership[:, i]
        dfc = df.iloc[members == 1]
        centroid = []
        for column in columns:
            centroid.append(dfc[column].mean())
        centroids.append(centroid)

    return centroids
