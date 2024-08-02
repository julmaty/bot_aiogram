FROM python:3.12.4-alpine

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Установим директорию для работы

WORKDIR /telegram_bot

COPY ./requirements.txt ./

# Устанавливаем зависимости и gunicorn
RUN pip install pip==24.0
RUN pip index versions faiss-cpu
RUN pip install --no-cache-dir -r ./requirements.txt

# Копируем файлы и билд
COPY ./ ./

RUN chmod -R 777 ./