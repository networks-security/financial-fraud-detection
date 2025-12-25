import datetime
from dateparser import parse
from ml_service.core.schemas.transaction_model import TransactionModel
from ml_service.application.api.dtos.transaction_dto import TransactionRequestBodyDto


def transaction_request_dto_to_model(
    dto: TransactionRequestBodyDto,
) -> TransactionModel:
    return TransactionModel(
        request_id=dto.request_id,
        transaction_id=dto.transaction_id,
        tx_datetime=parse(dto.tx_datetime) or datetime.datetime.now(),
        customer_id=dto.customer_id,
        terminal_id=dto.terminal_id,
        tx_amount=dto.tx_amount,
        tx_time_seconds=dto.tx_time_seconds,
        tx_time_days=dto.tx_time_days,
    )
