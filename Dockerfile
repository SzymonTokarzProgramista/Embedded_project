#------------------- Build Stage -------------------
FROM python:3.11-slim AS build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV PATH="/venv/bin:$PATH"
RUN python -m venv /venv

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#------------------- Final Stage -------------------
FROM python:3.11-slim

ENV PATH="/venv/bin:$PATH"

# Copy virtual environment from build stage
COPY --from=build /venv /venv

# Set working directory and copy project files
WORKDIR /Embedded_project
COPY . .

# Expose port and run application
EXPOSE 5000
CMD ["python", "-u", "app.py"]