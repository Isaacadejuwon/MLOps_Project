# 1. Use an official, lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this leverages Docker layer caching)
COPY requirements.txt .

# 4. Install the Python dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and the model_dir into the container
COPY . .

# 6. Expose the port FastAPI uses
EXPOSE 8000

# 7. Start the Uvicorn web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]