FROM ubuntu:20.04
MAINTAINER Changjae Lee <changjae@buffalo.edu>

# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG VCS_REF
ARG VCS_URL
ARG VERSION=0.1
ARG ARCH=amd64
ARG FRAMEWORK_VERSION
ARG GCC_VERSION=9
ARG TENSORFLOW_VERSION=1.14.0

######################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends \
  build-essential \
  software-properties-common && \
  add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
  apt-get update && apt-get install -y --no-install-recommends \
  gcc-${GCC_VERSION} \
  g++-${GCC_VERSION}  \
  libvips          \
  libjpeg-turbo8-dev \
  libturbojpeg \
  libvips-dev \
  libvips42 \
  libgl1 \
  build-essential \
  pkg-config \
  git \
  libopenblas-dev \
  libopenblas-base \
  python3-dev \
  libcurl4-openssl-dev \
  libgtest-dev \
  cmake \
  curl \
  wget \
  unzip \
  apt-transport-https \
  ca-certificates \
  bison \
  tzdata \
  locales \
  subversion \
  && \
  rm -rf /var/lib/apt/lists/* && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 60 --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} && \
  update-alternatives --config gcc && \
  gcc --version

RUN ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

########## DEPENDENCIES INSTALLATION ####################
RUN apt update && apt install -y --no-install-recommends \
  python3-dev \
  python3-pip \
  python3-numpy \
  ninja-build \
  libgoogle-glog-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install typing_extensions pyyaml dataclasses

########## ANACONDA INSTALLATION ########################
RUN mkdir -p /miniconda3 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O /miniconda3/miniconda.sh 
RUN bash /miniconda3/miniconda.sh -b -u -p /miniconda3 
RUN rm -rf /miniconda3/miniconda.sh 
RUN /miniconda3/bin/conda init bash 
ENV PATH $PATH:/miniconda3/bin/

########## ANACONDA ENV ACTIVATION ###################### 
RUN conda create -n tf114gpu 
RUN echo "source activate tf114gpu" > ~/.bashrc
ENV PATH /opt/conda/envs/tf114gpu/bin:$PATH
# RUN conda activate tf114gpu 
SHELL ["conda", "run", "-n", "tf114gpu", "/bin/bash", "-c"]
RUN conda install -c anaconda tensorflow-gpu=1.14.0 

########## PY-MLMODELSCOPE INSTALLATION #################
RUN pip3 install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio==1.27.2 
RUN pip3 install google-pasta 
RUN pip3 install opencv-python 
RUN pip3 install Pillow 
RUN pip3 install aenum 
RUN conda install -c pytorch torchvision 
RUN conda install -c conda-forge onnx 
RUN conda install -c anaconda scipy 
RUN pip3 install psycopg 
RUN pip3 install "psycopg[binary]" 
RUN pip3 install Pika 
RUN pip3 install requests 
RUN git clone https://github.com/xlab-ub/py-mlmodelscope.git 
RUN cd ./py-mlmodelscope 
# RUN python3 --version 
WORKDIR /py-mlmodelscope 
ENV CONDA_DEFAULT_ENV tf114gpu 
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf114gpu", "python3", "run_mlmodelscope.py", "--standalone", "false", "--agent", "tensorflow", "--env_file", "true"] 
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf114gpu", "python3"] 