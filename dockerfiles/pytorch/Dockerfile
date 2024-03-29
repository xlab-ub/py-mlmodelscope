FROM ubuntu:18.04
MAINTAINER Changjae Lee <changjae@buffalo.edu>

# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG VCS_REF
ARG VCS_URL
ARG VERSION=0.1
ARG ARCH=amd64
# ARG FRAMEWORK_VERSION
# ARG GCC_VERSION=9
ARG PYTORCH_VERSION=1.8.1

ENV PYTHON_VERSION=3.8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH /opt/anaconda3/bin:$PATH

######################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  git \
  build-essential \
  software-properties-common \
  ca-certificates \
  wget \
  curl \
  htop \
  zip \
  unzip \
  tzdata \
  locales && \
  rm -rf /var/lib/apt/lists/* 

RUN ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

########## ANACONDA INSTALLATION ########################
RUN cd /opt && \
  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O miniconda.sh && \
  /bin/bash ./miniconda.sh -b -p /opt/anaconda3 && \
  rm miniconda.sh && \
  /opt/anaconda3/bin/conda clean --all -y -f && \
  ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc && \
  conda config --set always_yes yes --set changeps1 no
RUN pip install --upgrade pip

########## PYTORCH INSTALLATION #########################
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 

########## PY-MLMODELSCOPE INSTALLATION #################
RUN pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
RUN pip install opencv-python-headless
RUN pip install aenum requests
RUN pip install psycopg "psycopg[binary]"
RUN pip install Pika 
RUN cd /
RUN git clone https://github.com/xlab-ub/py-mlmodelscope.git 
RUN cd ./py-mlmodelscope 
WORKDIR /py-mlmodelscope 
ENTRYPOINT ["conda", "run", "--no-capture-output", "python", "run_mlmodelscope.py", "--standalone", "false", "--agent", "pytorch", "--env_file", "true"] 