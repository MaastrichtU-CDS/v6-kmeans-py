# -*- coding: utf-8 -*-

""" Sample code to test the federated algorithm with a mock client
"""
import os
from vantage6.tools.mock_client import ClientMockProtocol


# Start mock client
data_dir = os.path.join(os.getcwd(), 'v6_kmeans_py', 'local')
client = ClientMockProtocol(
    datasets=[
        os.path.join(data_dir, 'data1.csv'),
        os.path.join(data_dir, 'data2.csv')
    ],
    module='v6_kmeans_py'
)

# Get mock organisations
organizations = client.get_organizations_in_my_collaboration()
print(organizations)
ids = [organization['id'] for organization in organizations]

# # Check partial method
# partial_task = client.create_new_task(
#     input_={
#         'method': 'initialize_centroids_partial',
#         'kwargs': {
#             'k': 3,
#             'columns': [
#                 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
#             ]
#         }
#     },
#     organization_ids=ids
# )
# results = client.get_results(partial_task.get('id'))
# print(results)

# Check master method
master_task = client.create_new_task(
    input_={
        'master': True,
        'method': 'master',
        'kwargs': {
            'org_ids': [0, 1],
            'k': 3,
            'columns': [
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ]
        }
    },
    organization_ids=[0, 1]
)
results = client.get_results(master_task.get('id'))
print(results)
