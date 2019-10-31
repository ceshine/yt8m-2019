FROM nvidia/cuda:10.1-base-ubuntu18.04

LABEL maintainer="ceshine@ceshine.net"

ARG PYTHON_VERSION=3.7.4
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME

ADD environment.yml /tmp
WORKDIR /tmp
RUN conda init bash
RUN conda env create -f /tmp/environment.yml && conda clean -a -y

ADD . /home/$USERNAME/src
RUN sudo chown $USERNAME /home/$USERNAME/src -R
WORKDIR /home/$USERNAME/src

RUN echo "source activate yt8m" > ~/.bashrc
RUN bash -c "source activate yt8m && pip install PyTorchHelperBot/."
RUN bash -c "source activate yt8m && pip install pyyaml"

RUN bash