Create web server image
`docker build -f ml_service/infra/docker/uvicorn/Dockerfile -t ml-api:1.0`

Run the container
`docker run ml-api:1.0`

You can use these transaction details to test POST localhost:8000/transaction/analyze

```
{
  "request_id": "REQ-100001",
  "transaction_id": 987654,
  "tx_datetime": "2025-12-21T09:15:32Z",
  "customer_id": 1023,
  "terminal_id": 12,
  "tx_amount": 45,
  "tx_time_seconds": 33332,
  "tx_time_days": 19654
}
```
