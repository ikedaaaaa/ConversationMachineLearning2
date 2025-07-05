FROM python:3.11

WORKDIR /app

RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --allow-unauthenticated gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# RUN mkdir -p models

# EXPOSE 8888

# CMD ["python", "main_advanced.py"]
