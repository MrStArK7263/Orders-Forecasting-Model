FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model.pkl ./

ENV MODEL_PATH=model.pkl
EXPOSE 8000

# Render gives the port in $PORT; locally it falls back to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
