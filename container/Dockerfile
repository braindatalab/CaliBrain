# CaliBrain without a display, with Jupyter: the released package from PyPI on a slim Python.
# No example data is inside the image; mount a folder and pass it as data_dir.
#
#   docker build -f container/Dockerfile --build-arg VERSION=1.1.0 -t calibrain container
#
# VERSION is a release number on PyPI. "local" (the default) installs the wheel in
# container/wheels instead (used by the checks of pull requests).
# The build context is the folder container/. The Python version must match the one used
# by scripts/make_container_constraints.py and by container/calibrain.def.
FROM python:3.13-slim

ARG VERSION=local

LABEL org.opencontainers.image.title="CaliBrain" \
      org.opencontainers.image.description="CaliBrain: A Python toolbox for uncertainty quantification, calibration, and benchmarking in Bayesian M/EEG inverse source imaging" \
      org.opencontainers.image.source="https://github.com/braindatalab/CaliBrain" \
      org.opencontainers.image.documentation="https://calibrain.readthedocs.io/" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      org.opencontainers.image.version="${VERSION}"

COPY constraints.txt install_calibrain.sh /opt/calibrain/
COPY wheels/ /opt/calibrain/wheels/
RUN sh /opt/calibrain/install_calibrain.sh /opt/calibrain/constraints.txt "${VERSION}" /opt/calibrain/wheels \
    && rm -rf /opt/calibrain

# Matplotlib writes plots to files; there is no display.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# Docker runs as an ordinary user. Apptainer ignores this and runs as the calling user.
RUN useradd --create-home --uid 1000 calibrain \
    && mkdir /data /work \
    && chown calibrain:calibrain /data /work
USER calibrain
WORKDIR /work

CMD ["bash"]
