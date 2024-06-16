FROM python:3.11.9-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY backup-milvus.py .

CMD python backup-milvus.py