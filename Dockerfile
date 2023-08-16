# Basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:3.4.2


# Package name
ARG PKG_NAME="v6_kmeans_py"

# Install federated algorithm
COPY . /app
RUN pip install /app

# Tell docker to execute `docker_wrapper()` when the image is run.
ENV PKG_NAME=${PKG_NAME}
CMD python -c "from vantage6.tools.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
