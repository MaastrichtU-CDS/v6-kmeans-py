# -*- coding: utf-8 -*-

""" Federated algorithm for kmeans

We follow the approach introduced by Stallmann and Wilbik (2022),
but implementing it for classical kmeans.
"""
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.cluster import KMeans
from vantage6.tools.util import info
from v6_kmeans_py.helper import coordinate_task


def master(
        client, data: pd.DataFrame, k: int, epsilon: int = 0.05,
        max_iter: int = 300, columns: list = None, org_ids: list = None
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

    # Initialise k global cluster centres, for now start with k random points
    # drawn from a random data node
    info('Initializing k global cluster centres')
    input_ = {
        'method': 'initialize_centroids_partial',
        'kwargs': {'k': k, 'columns': columns}
    }
    results = coordinate_task(client, input_, [np.random.choice(ids)])
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

        # Organise local centroids into a matrix
        local_centroids = []
        for result in results:
            for local_centroid in result:
                local_centroids.append(local_centroid)
        X = np.array(local_centroids)

        # Average centroids by running kmeans on local results
        # TODO: add other averaging options
        kmeans = KMeans(
            n_clusters=k, random_state=0, n_init='auto',
        ).fit(X)
        new_centroids = kmeans.cluster_centers_

        # Compute the sum of the magnitudes of the centroids differences
        # between steps. This change in centroids between steps will be used
        # to evaluate convergence.
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
        data: pd.DataFrame, k: int, columns: list = None
) -> list:
    """ Initialise global centroids for kmeans

    Parameters
    ----------
    data
        Dataframe with input data
    k
        Number of clusters
    columns
        Columns to be used for clustering

    Returns
    -------
    centroids
        Initial guess for global centroids
    """
    # TODO: use a better method to initialize centroids
    info(f'Randomly sample {k} data points to use as initial centroids')
    if columns:
        df = data[columns].sample(k)
    else:
        df = data.sample(k)

    # Organise initial guess for centroids as a list
    centroids = []
    for index, row in df.iterrows():
        centroids.append(row.values.tolist())

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
