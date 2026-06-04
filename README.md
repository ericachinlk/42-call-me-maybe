# only if uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install Python version managed by uv (safe step)
uv python install

# IMPORTANT: avoid filling /home disk
mkdir -p /tmp/uv-cache /tmp/call-me-maybe-venv

export UV_CACHE_DIR=/tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv

make install / uv sync

# clean uninstall
make clean
rm -rf /tmp/uv-cache
rm -rf /tmp/call-me-maybe-venv
these should be removed after reboot

First execution may take longer due to model download and initialization.
Subsequent runs are significantly faster due to caching.