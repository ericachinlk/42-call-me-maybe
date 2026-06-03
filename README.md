# only if uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install Python version managed by uv (safe step)
uv python install

# IMPORTANT: avoid filling /home disk
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv

uv sync