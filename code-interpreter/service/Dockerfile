# Control-plane service: FastAPI app that manages executor containers
# (docker-out-of-docker). It never runs user Python itself.
FROM python:3.11-slim

# dockerd + docker CLI support the docker-in-docker entrypoint mode; the
# docker-out-of-docker mode only needs the CLI against a mounted socket.
# (Debian trixie ships the CLI as a separate package.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        docker.io \
        docker-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md entrypoint.sh ./
COPY app ./app

RUN chmod +x entrypoint.sh && pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["code-interpreter-api"]
