# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 1. cuda11.8 base with pytorch11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS pyt2.2.1-cu11.8

RUN apt-get update && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
  python3 \
  python3-pip \
  python3-dev \
  python3-setuptools \
  sudo \
  && rm -rf /var/lib/apt/lists/*

ARG UNAME=dev
ARG UID=1000
ARG GID=1000

# Add normal sudo-user to container, passwordless
RUN addgroup --gid $GID $UNAME \
  && adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME \
  && adduser $UNAME sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
  && sed -i 's/required/sufficient/' /etc/pam.d/chsh \
  && touch /home/$UNAME/.sudo_as_admin_successful

WORKDIR /home/${UNAME}/
ENV HOME=/home/${UNAME}
USER ${UNAME}
ENV PATH="${PATH}:${HOME}/.local/bin"

# pytorch via wheel
RUN pip install --no-cache-dir \
  pip \
  setuptools \
  "numpy<=1.25" \
  torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cuml-cu11==24.2.*" 

# 2. minkowski intermediate stage
FROM pyt2.2.1-cu11.8 AS minkowski-pyt2.2.1-cu11.8

RUN sudo apt-get update && sudo apt-get install --no-install-recommends -y \
  ninja-build \
  libopenblas-dev \
  cmake \
  build-essential \
  git \
  && sudo rm -rf /var/lib/apt/lists/*

# From the minkowski repo: You should modify this to match your GPU compute capability
# You can check what to fill here based on the output of nvcc --list-gpu-arch
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=8
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python3 setup.py install --user --force_cuda --blas=openblas

# 3. Digiforests devkit, final stage
FROM minkowski-pyt2.2.1-cu11.8 AS digiforests_devkit

RUN sudo apt-get update && sudo apt-get install --no-install-recommends -y \
  # open3d system dependencies: https://www.open3d.org/docs/release/docker.html
  libegl1 \
  libgl1 \
  libgomp1 \
  # following can be removed later
  zsh \
  tmux \
  neovim \
  tree \
  curl \
  wget \
  && sudo rm -rf /var/lib/apt/lists/*

# once it gets merged in, use the master
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
RUN sudo sh -c "$(curl -L https://raw.githubusercontent.com/balancedscorpion/zsh-in-docker/new_user/zsh-in-docker.sh)" -- \
  -u ${UNAME} \
  -x \
  -p git -p z -p sudo -p history -p copybuffer -p copypath -p copyfile \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions

WORKDIR /home/${UNAME}/digiforests_devkit
COPY --chown=${UID}:${GID} . /home/${UNAME}/digiforests_devkit
RUN pip install --no-cache-dir -v -e ".[test]"
# RUN pytest -v tests

SHELL ["/bin/zsh", "-lc"]
CMD ["/bin/zsh"]
