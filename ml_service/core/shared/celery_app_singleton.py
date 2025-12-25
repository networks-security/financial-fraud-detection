import celery

from ml_service.infra.web_server.config import CeleryTasksConfig


class CeleryAppSingleton:
    """
    Manages a single instance of Celery app throughout the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CeleryAppSingleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def __init__(self) -> None:
        self._initialize()

    def _initialize(self):
        self.task_celery = CeleryTasksConfig()
        self.celery_app = celery.Celery("ml_service")
        self.celery_app.config_from_object(self.task_celery)
        # self.celery_app.autodiscover_tasks([CeleryTasksConfig.task_name])

    def get_celery_app(self):
        return self.celery_app
