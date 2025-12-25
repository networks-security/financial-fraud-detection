import celery
import celery.result

from ml_service.core.schemas.transaction_model import TransactionModel
from ml_service.core.shared.celery_app_singleton import CeleryAppSingleton
from ml_service.infra.web_server.config import CeleryTasksConfig


class PutTransactionInQueueService:
    def __init__(self):
        pass

    def execute(self, transaction: TransactionModel):
        print("Transaction is being enqueued...")

        celery_app = CeleryAppSingleton().get_celery_app()

        print("Transaction is being enqueued... Got the celery app...")

        celery_result: celery.result.AsyncResult = celery_app.send_task(
            CeleryTasksConfig.task_name,
            args=[transaction.model_dump()],
            queue=CeleryTasksConfig.task_queue,
        )
        print("Transaction is being enqueued... Task sent with result ", celery_result)
        print("Enqueued transaction with task id: ", celery_result.id)

        # Return the task UUID for tracking purposes
        return celery_result.id
