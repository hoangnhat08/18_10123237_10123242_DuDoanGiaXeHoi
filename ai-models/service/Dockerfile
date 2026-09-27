# Build context = ./ai-models (xem docker-compose.yml o goc repo)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY service/ /app/service
COPY models/ /app/models
ENV AI_SERVICE_PORT=8001
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"
CMD ["python", "service/app.py"]
