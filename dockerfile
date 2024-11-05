FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8005

ENV PYTHONPATH "${PYTHONPATH}:/code"

CMD ["python", "webb/app.py"]
