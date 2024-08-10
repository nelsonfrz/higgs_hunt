FROM quay.io/jupyter/minimal-notebook:latest

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
    build-essential \
    # for latex labels
    cm-super \
    dvipng \
    # for matplotlib anim
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install Python 3 packages
RUN mamba install --yes \
    'conda-forge::blas=*=openblas' \
    'cython' \
    'ipympl'\
    'ipywidgets' \
    'jupyterlab-git' \
    'matplotlib-base' \
    'pandas' \
    'uproot' \
    'python-lsp-server' \
    'jupyterlab_code_formatter' \
    'black' \
    'isort' \
    'widgetsnbextension' && \
    mamba clean --all -f -y

COPY analysis/ .

USER ${NB_UID}

WORKDIR "${HOME}"

