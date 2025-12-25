# Use official Python image
FROM python:3.10-slim

# Add a non-root user
RUN adduser --disabled-password appuser
USER appuser

# Set working directory
WORKDIR /usr/src/app/

# Copy required files
COPY ml_service/infra/docker/uvicorn/requirements.txt ml_service/infra/docker/uvicorn/requirements.txt
COPY ml_service/infra/web_server ml_service/infra/web_server
COPY ml_service/application ml_service/application
COPY ml_service/core ml_service/core

# Create a virtual environment and install dependencies
RUN python -m venv ./venv
RUN . venv/bin/activate && \
    pip install -r ml_service/infra/docker/uvicorn/requirements.txt

# Set environment variables
ENV CELERY_BROKER_URL=redis://localhost:6379
ENV CELERY_RESULT_BACKEND=redis://localhost:6379
ENV PATH="/usr/src/app/venv/bin:$PATH"

# Expose port for FastAPI
EXPOSE 8000

# Default command to start the gateway server
CMD ["uvicorn", "ml_service.application.api.controllers.transaction_controller:app", "--host", "0.0.0.0", "--port", "8000"]
