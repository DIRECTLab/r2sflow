FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ====================================================
# fundamentals
# ====================================================

# The base image ships an apt source for NVIDIA's CUDA repo, but nothing below
# comes from it -- every package resolves from the Ubuntu archives. That repo sets
# Acquire-By-Hash: no and its CDN caches InRelease and Packages.gz independently,
# so whenever NVIDIA republishes the index a build can fetch a stale InRelease
# against a fresh Packages.gz and `apt-get update` dies on the size mismatch.
# Drop the source to take the whole failure mode out of the build. (Re-add it if
# you ever need to apt-get install a CUDA package inside the container.)
# Kept as one layer so the index is never cached separately from the install.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl fzf git htop sudo tmux tree vim wget zsh \
        ninja-build libsparsehash-dev ffmpeg \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm --recursive --force /var/lib/apt/lists/*

# ====================================================
# user setup
# ====================================================

ARG UNAME=docker
ARG UID=1000
ARG GID=1000

RUN groupadd --gid ${GID} ${UNAME} && \
    useradd --uid ${UID} --gid ${GID} --create-home --groups sudo ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${UNAME} \
    && chmod 440 /etc/sudoers.d/${UNAME}

USER ${UNAME}
ENV HOME="/home/${UNAME}"
WORKDIR ${HOME}

# ====================================================
# python env
# ====================================================

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/
RUN uv venv --python 3.10
ENV VIRTUAL_ENV="${HOME}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA="1"
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv pip install -r requirements.txt
RUN uv pip install git+https://github.com/mit-han-lab/torchsparse.git@v2.0.0 --no-build-isolation
# The upstream index (https://shi-labs.com/natten/wheels/) is unusable: its TLS
# certificate has expired, so uv/pip refuse to fetch from it. NATTEN's current
# index at whl.natten.org only publishes 0.17.5+ against torch 2.5.0+, which is
# too new for the 0.17-era natten.functional API that r2flow/models/hdit.py calls.
# This is the same artifact the dead index resolved to -- the wheel is attached to
# the upstream GitHub release. Pinned to cp310 / torch 2.1 / cu121 to match the
# venv above, the torch in requirements.txt, and this base image's CUDA.
RUN uv pip install --no-build-isolation \
    "https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.1/natten-0.17.1%2Btorch210cu121-cp310-cp310-linux_x86_64.whl"

WORKDIR ${HOME}/workspace