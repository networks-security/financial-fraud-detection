from pydantic import Field
from uuid import uuid4
from pydantic import BaseModel

class TransactionRequestBodyDto(BaseModel):
  request_id: str = Field(default_factory=lambda: str(uuid4()))
  transaction_id: int
  tx_datetime: str
  customer_id: int
  terminal_id: int
  tx_amount: int
  tx_time_seconds: int
  tx_time_days: int

class TransactionResponseBodyDto(BaseModel):
  response_id: str = Field(default_factory=lambda: str(uuid4()))