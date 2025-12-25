Create docker image:
`docker build -f ml_service/infra/docker/celery/Dockerfile -t celery-worker:1.0 .` (use `--no-cache` to disable automatic caching)

Run the container:
`docker run celery-worker:1.0`
