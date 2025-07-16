# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords')"

# Expose Flask on 5001
EXPOSE 5001

# Expose MLflow on 5000
EXPOSE 5000

# Install gunicorn for production-grade Flask serving (optional)
RUN pip install gunicorn

# Final CMD — MLflow on 5000, Flask on 5001
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & python app/app.py"]
