from datetime import datetime
from pydantic import BaseModel


class TransactionModel(BaseModel):
    request_id: str
    transaction_id: int
    tx_datetime: datetime
    customer_id: int
    terminal_id: int
    tx_amount: int
    tx_time_seconds: int
    tx_time_days: int
