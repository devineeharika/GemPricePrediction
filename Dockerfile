FROM python:3.9-slim

# Install necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    wget \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN wget -qO- https://sh.rustup.rs | sh -s -- -y

# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "application.py"]
