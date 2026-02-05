# Используем легкий Python 3.10
FROM python:3.10-slim

# Устанавливаем системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая папка внутри контейнера
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порты (8000 для бэка, 8501 для фронта)
EXPOSE 8000 8501

# Команда по умолчанию (будет переопределена в docker-compose)
CMD ["bash"]