FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY dashboard/requirements.txt /app/dashboard/requirements.txt

RUN pip install --no-cache-dir -r /app/dashboard/requirements.txt && pip install --no-cache-dir joblib scikit-learn

COPY dashboard/ /app/dashboard/
COPY ml/ /app/ml/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]