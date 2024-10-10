FROM python:3.11
USER root

WORKDIR /ragflow

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is available in the PATH
ENV POETRY_HOME="/root/.local"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Check if Poetry is installed correctly (for debugging purposes)
RUN poetry --version

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock* /ragflow/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Install additional packages
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --fix-missing nodejs nginx ffmpeg libsm6 libxext6 libgl1

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files
ADD ./api ./api
ADD ./conf ./conf
ADD ./deepdoc ./deepdoc
ADD ./rag ./rag
ADD ./agent ./agent
ADD ./graphrag ./graphrag

ENV PYTHONPATH=/ragflow/
ENV HF_ENDPOINT=https://hf-mirror.com

# Copy entrypoint script and make it executable
ADD docker/entrypoint.sh ./entrypoint.sh
ADD docker/.env ./
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
