# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Create a log directory and set permissions
RUN mkdir -p /var/log/app && chmod -R 777 /var/log/app

# Expose the port on which the API will run
EXPOSE 8000

# Command to run the application using Gunicorn (a production-grade web server)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]