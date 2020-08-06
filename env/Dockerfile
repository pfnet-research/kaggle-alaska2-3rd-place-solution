FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="Kaizaburo Chubachi <kaizaburo@preferred.jp>"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    htop \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    cmake \
    parallel \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
    emacs \
    tmux \
    language-pack-ja-base \
    language-pack-ja \
    zsh \
    less \
    gawk

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# clean up apt
RUN apt-get clean -y && \
    apt-get autoremove -y && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# --- python enviroments ---
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# prepend pytorch and conda-forge before default channel
RUN conda update -n base -c defaults conda && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch

COPY conda_env.yaml ${HOME}/conda_env.yaml
RUN conda env create -f ${HOME}/conda_env.yaml
ENV PATH /opt/conda/envs/alaska2/bin:$PATH

RUN export NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter labextension install jupyterlab-plotly --no-build && \
    jupyter labextension install plotlywidget --no-build && \
    jupyter lab build
RUN unset NODE_OPTIONS

RUN cd ~/ &&\
    git clone https://github.com/NVIDIA/apex.git &&\
    cd apex &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN wget https://github.com/dwgoon/jpegio/raw/master/dist/jpegio-0.2.1-cp37-cp37m-linux_x86_64.whl && \
    pip install jpegio-0.2.1-cp37-cp37m-linux_x86_64.whl

# Clean up conda
RUN conda clean -p && \
    conda clean -t && \
    conda clean --yes --all

# Clean up pip
RUN rm -rf $HOME/.cache
