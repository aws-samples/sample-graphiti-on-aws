# App image: Streamlit UI (default CMD) and MCP server (python mcp_server.py).
# Build where docker.io / pypi are reachable, or point the build args at mirrors:
#   --build-arg BASE_IMAGE=<ecr-cn>/mirror/python:3.12-slim
#   --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY frameworks/ frameworks/
COPY tools/ tools/
COPY examples/ examples/
COPY scripts/smoke_test.py scripts/
COPY app.py mcp_server.py __init__.py ./

RUN useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
