FROM python:3.11-slim

ENV \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/.venv
ENV \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.6.1

WORKDIR /real-estate-price-prediction

COPY poetry.lock pyproject.toml ./

RUN apt-get update \
    && apt-get install -y \
    git \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "poetry==$POETRY_VERSION"

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV" \
    && . "$VIRTUAL_ENV"/bin/activate \
    && poetry install --no-root \
    && apt-get clean



COPY ./app/ ./app
COPY ./ml/ ./ml

ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 8080
CMD uvicorn app.main:app --host 0.0.0.0 --port 8080