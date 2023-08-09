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
ARG ONNXRUNTIME_VERSION=1.7.0

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
RUN conda create -n ort170 python=3.8 
RUN echo "source activate ort170" > ~/.bashrc
ENV PATH /opt/conda/envs/ort170/bin:$PATH
# RUN conda activate ort170 
SHELL ["conda", "run", "-n", "ort170", "/bin/bash", "-c"]
RUN pip3 install onnxruntime-gpu==1.7.0 
RUN conda install -c conda-forge cudatoolkit=11.0 cudnn 

########## PY-MLMODELSCOPE INSTALLATION #################
RUN pip3 install opencv-contrib-python 
RUN pip3 install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
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
ENV CONDA_DEFAULT_ENV ort170 
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ort170", "python3", "run_mlmodelscope.py", "--standalone", "false", "--agent", "onnxruntime", "--env_file", "true"] 
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ort170", "python3"] 