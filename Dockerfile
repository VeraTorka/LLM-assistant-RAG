FROM python:3.12-slim

WORKDIR /app

# Установим pipenv
RUN pip install --no-cache-dir pipenv

# Сначала копируем Pipfile/Lock и ставим зависимости
COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --ignore-pipfile --system

# Потом копируем данные и код
COPY data/ ./data/
COPY nutrition_assistant/ ./nutrition_assistant/

# Настройки Streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Проброс пути к данным
ENV DATA_PATH=/app/data/data.csv

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "nutrition_assistant/app.py"]

