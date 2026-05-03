FROM python:3.10-slim

WORKDIR /app

COPY fda-search-app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fda-search-app /app/fda-search-app

ENV OPENAI_API_KEY=""
EXPOSE 8501

CMD ["streamlit", "run", "fda-search-app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
