USERNAME="$(id -u -n)"
MODULE="tasks"
REPO_ROOT="repo"
LOGS_ROOT="${REPO_ROOT}/logs/tasks"
LOGS_PATH="${LOGS_ROOT}/daemon.log"

. ./venv/bin/activate && \
celery -A ml_service.tasks.tasks worker --loglevel=INFO -Q tasks && \
echo "Celery worker started successfully." && \
celery -A ml_service.tasks.tasks inspect registered