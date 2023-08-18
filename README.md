# Vantage6 algorithm for k-means

This algorithm was designed for the [vantage6](https://vantage6.ai/) 
architecture. 

## Input data

The data nodes should hold a `csv` with variables following the same common 
data model. We split the [iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
and provide as an example to test the code.

## Using the algorithm

Below you can see an example of how to run the algorithm:

``` python
import time
from vantage6.client import Client

# Initialise the client
client = Client('http://127.0.0.1', 5000, '/api')
client.authenticate('username', 'password')
client.setup_encryption(None)

# Define algorithm input
input_ = {
    'method': 'master',
    'master': True,
    'kwargs': {
        'org_ids': [2, 3],  # organisations to run kmeans
        'k': 3,             # number of clusters to compute
        'epsilon': 0.05,    # threshold for convergence criterion
        'max_iter': 300,    # maximum number of iterations to perform
        'columns': [        # columns to be used for clustering
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ]
    }
}

# Send the task to the central server
task = client.task.create(
    collaboration=1,
    organizations=[2, 3],
    name='v6-kmeans-py',
    image='aiaragomes/v6-kmeans-py:latest',
    description='run kmeans',
    input=input_,
    data_format='json'
)

# Retrieve the results
task_info = client.task.get(task['id'], include_results=True)
while not task_info.get('complete'):
    task_info = client.task.get(task['id'], include_results=True)
    time.sleep(1)
result_info = client.result.list(task=task_info['id'])
results = result_info['data'][0]['result']
```

## Testing locally

If you wish to test the algorithm locally, you can create a Python virtual 
environment, using your favourite method, and do the following:

``` bash
source .venv/bin/activate
pip install -e .
python v6_kmeans_py/example.py
```

The algorithm was developed and tested with Python 3.7.

## Acknowledgments

This project was financially supported by the
[AiNed foundation](https://ained.nl/over-ained/).
