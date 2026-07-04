FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expect apps to run on port 7860
EXPOSE 7860

# Start the Flask app using Gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
