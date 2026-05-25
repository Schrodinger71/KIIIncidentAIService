FROM python:3.11-slim

WORKDIR /app

COPY inference-api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference-api/app.py ./app.py
COPY kii_methodology.py ./kii_methodology.py
COPY inference-api/models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
