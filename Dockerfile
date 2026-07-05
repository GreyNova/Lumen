FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by PyTorch and SentenceTransformers
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt

# Create a user with UID 1000 (Hugging Face standard)
RUN useradd -m -u 1000 user

# Copy the rest of the application
COPY . .

# Change ownership to the new user so SQLite can write to chroma_db
RUN chown -R user:user /app
USER user

# Hugging Face Spaces expect apps to run on port 7860
EXPOSE 7860

# Start the Flask app using Gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
