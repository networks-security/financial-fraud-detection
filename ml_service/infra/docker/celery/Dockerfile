# Use official Python image
FROM python:3.10-slim

# Install libgomp for LightGBM (for Celery workers)
RUN apt-get update && apt-get install -y libgomp1

# Add a non-root user
RUN adduser --disabled-password appuser
USER appuser

# Set working directory
WORKDIR /usr/src/app/

# Create a virtual environment and install dependencies
RUN python -m venv ./venv
COPY ml_service/infra/docker/celery/requirements.txt ml_service/infra/docker/celery/requirements.txt
RUN . venv/bin/activate && \
    pip install --no-cache-dir -r ml_service/infra/docker/celery/requirements.txt

# Copy application code
# TODO: Optimize Dockerfile to copy only necessary files
COPY ml_service ml_service

# Set environment variables
ENV CELERY_BROKER_URL=redis://localhost:6379
ENV CELERY_RESULT_BACKEND=redis://localhost:6379
ENV PATH="/usr/src/app/venv/bin:$PATH"

EXPOSE 6379

# Entrypoint for Celery worker
ENTRYPOINT ["sh", "ml_service/infra/docker/celery/entrypoint.sh"]