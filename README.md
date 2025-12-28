# Real-time ML-powered web system for detecting financial fraud in banking

## Microservices Architecture

- **Frontend**: A Next.js (TypeScript) dashboard that displays transaction status, real-time fraud alerts, and system metrics.

- **Gateway**: An Express.js (TypeScript) backend that serves as the system’s entry point, handling client requests and routing them to the appropriate services.

- **ML API Service**: A FastAPI service that receives transactions from the Gateway, and enqueues tasks into Redis for asynchronous ML processing. This service does not perform ML inference directly.

- **Redis Queue**: An in-memory queue and message broker that decouples the ML API from the workers.

- **Multiple Celery Workers**: Execute the actual ML pipeline, consuming tasks from Redis, and performing model inference (~200ms per transaction).

- **Data Persistence Service**: A FastAPI service responsible for encrypting and storing transaction data in Firebase.

## Scalability and Performance

The ML Service is configured with Celery workers and a Redis queue, enabling the system to scale both horizontally and vertically. Redis and workers will run in separate containers on the server. This setup allows the system to process thousands of transactions per minute, depending on the hardware capabilities. The ML pipeline analyzes each transaction in ~200ms.

## Dockerized Services

Currently, only the ML Service has Dockerfiles configured. You can run it using Docker Compose:

```bash
DOCKER_BUILDKIT=1 docker compose --file ml_service/infra/docker/compose/docker-compose.yml -p ml-service up --build
```

```bash
docker-compose up
```

### Docker Status

- [ ] Frontend
- [ ] Gateway
- [x] ML API Service
- [x] Redis Queue
- [x] Multiple Celery Workers
- [ ] Data Persistence Service

## Deployment

The system will soon be deployed on Azure Cloud with CI/CD configured via GitHub Actions.

## Learn More

For more details about each service, refer to their respective `README.md` files. Some module subdirectories also contain setup instructions and usage guidelines.
