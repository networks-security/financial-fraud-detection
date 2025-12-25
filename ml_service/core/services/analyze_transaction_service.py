from ml_service.core.schemas.transaction_model import TransactionModel
from ml_service.ml_pipeline.detect_fraud import detect_fraud


class AnalyzeTransactionService:
    def process(self, transaction: TransactionModel):
        return detect_fraud(transaction)
