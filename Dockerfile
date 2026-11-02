FROM python:3.12-slim
WORKDIR /app
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser app.py .
COPY --chown=appuser src ./src
COPY --chown=appuser assets ./assets
COPY --chown=appuser demo ./demo
USER appuser
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:server"]
