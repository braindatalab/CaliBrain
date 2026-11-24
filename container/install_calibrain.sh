#!/bin/sh
# Installs CaliBrain and Jupyter into the image with the pinned package versions.
# Used by both container/Dockerfile and container/calibrain.def, so they cannot differ.
#
#   install_calibrain.sh CONSTRAINTS VERSION WHEEL_DIR
#
# VERSION is a release number on PyPI, or "local" to install the wheel found in WHEEL_DIR
# (the pull request check builds it from the checkout).
set -eu

constraints="$1"
version="$2"
wheel_dir="$3"

if [ "$version" = "local" ]; then
    wheel=$(ls "$wheel_dir"/calibrain-*.whl 2>/dev/null | head -n 1 || true)
    if [ -z "$wheel" ]; then
        echo "VERSION=local needs a calibrain-*.whl in $wheel_dir" >&2
        exit 1
    fi
    target="$wheel"
else
    target="calibrain==$version"
fi

python -m pip install --no-cache-dir -c "$constraints" "$target" notebook
