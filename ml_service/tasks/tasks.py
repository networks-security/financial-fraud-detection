import sys
import os


from ml_service.core.services.analyze_transaction_service import (
    AnalyzeTransactionService,
)
from ml_service.core.schemas.transaction_model import TransactionModel
from ml_service.core.shared.celery_app_singleton import CeleryAppSingleton

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

app = CeleryAppSingleton().get_celery_app()


@app.task(time_limit=5, name="tasks.tasks.analyze_transaction")
def analyze_transaction(transaction: TransactionModel):
    print(f"Analyzing transaction: {transaction}")
    result = AnalyzeTransactionService().process(transaction)
    print(f"Analysis finished, result: {result}")
    return result
