from fastapi import FastAPI

from ml_service.application.api.dtos.transaction_dto import TransactionRequestBodyDto
from ml_service.application.api.mappers.transaction_mapper import (
    transaction_request_dto_to_model,
)
from ml_service.core.services.put_transaction_in_queue_service import (
    PutTransactionInQueueService,
)

app = FastAPI()


@app.post("/transaction/analyze")
def transaction_analyze(transaction: TransactionRequestBodyDto):

    print("Received transaction for analysis: ", transaction)

    # Transform DTO into Model
    transaction_model = transaction_request_dto_to_model(transaction)

    print("Transformed transaction dto --> model: ", transaction_model)

    # Put the transaction model in Redis queue
    task_id = PutTransactionInQueueService().execute(transaction_model)

    print("Enqueued transaction with task id: ", task_id)

    if task_id is None:
        return {
            "status": "error",
            "code": 500,
            "message": "Failed to enqueue transaction",
            "task_id": task_id,
        }

    return {
        "status": "success",
        "code": 200,
        "message": "Transaction successfully enqueued",
        "task_id": task_id,
    }
