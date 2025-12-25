import os


class CeleryTasksConfig:
    task_queue = "tasks"
    task_name = "tasks.tasks.analyze_transaction"
    broker_url = os.environ.get(
        "CELERY_BROKER_URL", "redis://localhost:6379"
    )  # NOTE: Defined in Docker Compose web service
    result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
    worker_prefetch_multiplier = int(
        os.environ.get("CELERY_WORKER_PREFETCH_MULTIPLIER", 1)
    )
