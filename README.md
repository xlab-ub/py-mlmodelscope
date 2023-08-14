# py-mlmodelscope 

MLModelScope 

# [Documentation](https://docs.mlmodelscope.org/)

The current landscape of Machine Learning (ML) and Deep Learning (DL) is rife with non-uniform models, frameworks, and system stacks but lacks standard tools to evaluate and profile models or systems.
Due to the absence of such tools, the current practice for evaluating and comparing the benefits of proposed AI innovations (be it hardware or software) on end-to-end AI pipelines is both arduous and error prone --- stifling the adoption of the innovations.

MLModelScope is a hardware/software agnostic, extensible and customizable platform for evaluating and profiling ML models across datasets/frameworks/hardware, and within AI application pipelines.
MLModelScope lowers the cost and effort for performing model evaluation and profiling, making it easier for others to reproduce, evaluate, and analyze acurracy or performance claims of models and systems.

It is designed to aid in:

1. reproducing and comparing with published models, and designing models with performance and deployment in mind,
2. understanding the model performance (within realworld AI workflows) and its interaction with all levels of the hardware/software stack
3. discovering models, frameworks and hardware that are applicable to users' datasets.


To achieve this, MLModelScope:

- Provides a consistent evaluation, aggregation, and reporting system by defining
  - techniques to specify and provision workflows with HW/SW stacks
  - abstractions for evaluation and profiling using different frameworks
  - data consumption for evaluation outputs
- Enables profiling of experiments throughout the entire pipeline and at different abstraction levels (application, model, framework, layer, library and hardware, as shown on the right)
<!-- - Is framework and hardware agnostic - with current support for PyTorch, TensorFlow, ONNXRuntime, MXNet running on X86, PowerPC, and ARM CPU with GPU and FPGA --> 
- Is framework agnostic - with current support for PyTorch, TensorFlow, ONNXRuntime, MXNet 
- Is extensible and customizable - allowing users to extend MLModelScope by adding models, frameworks, or library and system profilers.
- Can run experiments on separate machines, and behind firewall (does not exposing model weights or machine specification)
- Allows parallel evaluation (multiple instantiations of the same experiment set-up across systems)
- Specifies model and framework resources as asset files which can be added easily, even at runtime


MLModelScope can be used as an application with a command line, API or web interface, or can be compiled into a standalone library. We also provide an online hub of continuously updated assets, evaluation results, and access to hardware resources — allowing users to discover and evaluate models without installing or configuring systems.

# Bare Minimum Installation

## Requirements 

``` 
python>=3.7
opentelemetry-api 
opentelemetry-sdk 
opentelemetry-exporter-otlp-proto-grpc 
grpcio 
``` 

## Prerequsite System Library Installation
We first discuss a bare minimum pytorch-agent installation without the tracing and profiling capabilities. To make this work, you will need to have the following system libraries preinstalled in your system.

- The CUDA library (required)
- The CUPTI library (required)
- The cuDNN library (not compulsory, required for mxnet-agent) 
- The Pytorch Python library (not compulsory, required for pytorch-agent) 
- The Tensorflow Python library (not compulsory, required for tensorflow-agent) 
- The ONNXRuntime and ONNX Python library (not compulsory, required for onnxruntime-agent) 
- The MXNet Python library (not compulsory, required for mxnet-agent) 

### The CUDA Library

Please refer to Nvidia CUDA library installation on this. Find the localation of your local CUDA installation, which is typically at `/usr/local/cuda/`, and setup the path to the `libcublas.so` library. 

### The CUPTI Library

Please refer to Nvidia CUPTI library installation on this. Find the localation of your local CUPTI installation, which is typically at `/usr/local/cuda/extras/CUPTI`, and setup the path to the `libcupti.so` library. 

Also, please install Pre-requsite Dynamic Library. 

**On Linux**

```bash
cd pycupti/csrc 
export PATH="/usr/local/cuda/bin:$PATH" 
nvcc -O3 --shared -Xcompiler -fPIC utils.cpp -o libutils.so -lcuda -lcudart -lcupti -lnvperf_host -lnvperf_target -I /usr/local/cuda/extras/CUPTI/include -L /usr/local/cuda/extras/CUPTI/lib64 
```

**On Windows**

```console
cd pycupti/csrc 
nvcc -O3 --shared utils.cpp -o utils.dll -I"%CUDA_PATH%/include" -I"%CUDA_PATH%/extras/CUPTI/include" -L"%CUDA_PATH%"/extras/CUPTI/lib64 -L"%CUDA_PATH%"/lib/x64 -lcuda -lcudart -lcupti -lnvperf_host -lnvperf_target -Xcompiler "/EHsc /GL /Gy /O2 /Zc:inline /fp:precise /D "_WINDLL" /Zc:forScope /Oi /MD" && del utils.lib utils.exp 
```

After running above commands, please check whether  `libutils.so` on Linux or `utils.dll` on Windows is in `pycupti/csrc` directory. 

### The Pytorch Python Library (not compulsory, required for pytorch-agent) 

The Pytorch Python library is required for our pytorch-agent. 

You can install Pytorch Python by referencing [Pytorch](https://pytorch.org/get-started/locally/). 

<details> 
<summary>PyTorch v1.8.1 with CUDA v11.1 Installation in Anaconda Environment</summary> 

## Anaconda Environment 

```bash 
conda create -n pytorch181cu111 python=3.8 
conda activate pytorch181cu111 
``` 

## PyTorch 

```bash 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
``` 

## OpenTelemetry 

```bash 
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
``` 

## OpenCV 

```bash 
pip install opencv-contrib-python 
``` 

## aenum 

```bash 
pip install aenum 
```

## requests 

```bash
pip install requests
``` 

## Psycopg (not compulsory, required for communicating with database) 

```bash
pip install psycopg
pip install "psycopg[binary]"
``` 

## Pika (not compulsory, required for communicating with messagequeue) 

```bash
pip install Pika 
``` 

</details>

### The Tensorflow Python Library (not compulsory, required for tensorflow-agent) 

The Tensorflow Python library is required for our tensorflow-agent. 

<details> 
<summary>Tensorflow v2 Installation in Anaconda Environment</summary> 

## Anaconda Environment 

```bash 
conda create -n tf2gpu python=3.8 
conda activate tf2gpu 
``` 

**On Windows** 

```bash 
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
# Anything above 2.10 is not supported on the GPU on Windows Native 
python -m pip install "tensorflow<2.11" 
# Verify install: 
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
``` 

## OpenCV 

```bash 
pip install opencv-contrib-python 
``` 

## OpenTelemetry 

```bash 
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
```

## aenum 

```bash 
pip install aenum 
``` 

## requests 

```bash
pip install requests
``` 

## Psycopg (not compulsory, required for communicating with database) 

```bash
pip install psycopg
pip install "psycopg[binary]"
``` 

## Pika (not compulsory, required for communicating with messagequeue) 

```bash
pip install Pika 
``` 

</details> 

<details> 
<summary>Tensorflow v1.14.0 with CUDA v10.0 Installation in Anaconda Environment</summary> 

## Anaconda Environment 

```bash 
conda create -n tf114gpu 
conda activate tf114gpu 
``` 

**On Windows** 

```bash 
conda install -c anaconda tensorflow-gpu=1.14.0 

# Verify install: 
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())" 
``` 

## OpenTelemetry 

```bash 
pip install opentelemetry-api 
pip install opentelemetry-sdk 
pip install opentelemetry-exporter-otlp-proto-grpc 
pip install grpcio==1.27.2 
pip install google-pasta # for tensorflow v1.14.0 
``` 

## OpenCV 

```bash 
pip install opencv-python # conda install -c conda-forge opencv 
``` 

## Pillow 

```bash 
pip install Pillow 
``` 

## aenum 

```bash 
pip install aenum 
``` 

## requests 

```bash
pip install requests
``` 

## Psycopg (not compulsory, required for communicating with database) 

```bash
pip install psycopg
pip install "psycopg[binary]"
``` 

## Pika (not compulsory, required for communicating with messagequeue) 

```bash
pip install Pika 
``` 

</details>

### The ONNXRuntime and ONNX Python Library (not compulsory, required for onnxruntime-agent) 

The ONNXRuntime and ONNX Python library is required for our onnxruntime-agent. 

<details> 
<summary>ONNXRuntime v1.7.0 Installation in Anaconda Environment</summary> 

## Anaconda Environment 

```bash 
conda create -n ort170 python=3.8 
conda activate ort170 
``` 

**On Windows and Linux** 

```bash
pip install onnxruntime-gpu==1.7.0 
conda install -c conda-forge cudatoolkit=11.0 cudnn 
# Verify install: 
# Even if the result is ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], it does not mean that GPU device can be used. 
python -c "import onnxruntime as ort;print(ort.get_available_providers())" 
``` 

## OpenTelemetry 

```bash 
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
``` 

## aenum 

```bash 
pip install aenum 
```

## TorchVision 

```bash
conda install -c pytorch torchvision 
```

## ONNX 

```bash 
conda install -c conda-forge onnx 
``` 

## Scipy 

```bash
conda install -c anaconda scipy 
```

## OpenCV 

```bash 
pip install opencv-python # conda install -c conda-forge opencv 
```

## requests 

```bash
pip install requests
``` 

## Psycopg (not compulsory, required for communicating with database) 

```bash
pip install psycopg
pip install "psycopg[binary]"
``` 

## Pika (not compulsory, required for communicating with messagequeue) 

```bash
pip install Pika 
``` 

</details>

### The MXNet Python Library (not compulsory, required for mxnet-agent) 

The MXNet Python library is required for our mxnet-agent. 

<details> 
<summary>MXNet v1.8.0 with CUDA v10.2 Installation in Anaconda Environment</summary> 

**v1.8.0 cannot be installed on Windows.** 

## cuDNN 

[cuDNN](https://developer.nvidia.com/rdp/cudnn-download) local installation is required for MXNet. 

Before issuing the following commands, you must replace X.Y and v8.x.x.x with your specific CUDA and cuDNN versions and package date. 

1. Navigate to your <cudnnpath> directory containing the cuDNN tar file.
2. Unzip the cuDNN package.

```bash
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz 
```

3. Copy the following files into the CUDA toolkit directory.

```bash
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* 
```

4. Edit the environment variables. 

```bash 
export PATH="/usr/local/cuda/bin:$PATH" 
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" 
```

**References** 

1. https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html 
2. https://github.com/tensorflow/tensorflow/issues/41041 

## Anaconda Environment 

```bash 
conda create -n mxnet180cu102 python=3.8 
conda activate mxnet180cu102 
``` 

**On Linux** 

v1.8.0 cannot be installed on Windows 

```bash
conda install -c anaconda cudatoolkit=10.2 cudnn=7.6.5 
conda install -c conda-forge libcblas # important 
pip install mxnet-cu102==1.8.0 
conda install -c conda-forge nccl 
python -m pip uninstall numpy 
python -m pip install numpy==1.23.1 

# Verify install: 
python -c "import mxnet as mx;print(mx.context.num_gpus())" 
``` 

## OpenTelemetry 

```bash 
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc grpcio 
``` 

## aenum 

```bash 
pip install aenum 
``` 

## TorchVision 

```bash
pip install torchvision==0.9.0 
```

## Scipy 

```bash
conda install -c anaconda scipy 
```

## Chardet 

```bash
pip install chardet 
```

## OpenCV 

```bash 
pip install opencv-contrib-python 
``` 

## requests 

```bash
pip install requests
``` 

## Psycopg (not compulsory, required for communicating with database) 

```bash
pip install psycopg
pip install "psycopg[binary]"
``` 

## Pika (not compulsory, required for communicating with messagequeue) 

```bash
pip install Pika 
``` 

</details>

## Test Installation

With the configuration and the above bare minimumn installation, you should be ready to test the installation and see how things works. 

To run an inference using the default DNN model `alexnet` with default test input images. 

```bash
python run_image_classification.py 
``` 

# External Service Installation to Enable Tracing and Profiling

We now discuss how to install a few external services that make the agent fully useful in terms of collecting tracing and profiling data.

## External Services

MLModelScope relies on a few external services. These services provide tracing functionality.

### Install OpenTelemetry library 

```bash 
pip install opentelemetry-api 
pip install opentelemetry-sdk 
pip install opentelemetry-exporter-otlp-proto-grpc 
pip install grpcio 
``` 

### Starting Trace Server

This service is required.

```bash 
docker run -d --name jaeger -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 -e COLLECTOR_OTLP_ENABLED=true -p 6831:6831/udp -p 6832:6832/udp -p 5778:5778 -p 16686:16686 -p 4317:4317 -p 4318:4318 -p 14250:14250 -p 14268:14268 -p 14269:14269 -p 9411:9411 jaegertracing/all-in-one:1.44 
``` 

The trace server runs on http://localhost:16686

# Use the system through Command Line 

An example run is 

```bash 
python run_image_classification.py --task image_classification --agent pytorch --model_name alexnet --architecture gpu --num_warmup 2 --dataset_name test --dataset_path ./test_data --batch_size 2
```

## Image Classification 

```bash 
python run_image_classification.py --task image_classification --agent pytorch --model_name alexnet --architecture gpu --num_warmup 2 --dataset_name test --batch_size 2 --gpu_trace false 
```

## Image Object Detection 

```bash 
python run_image_object_detection.py --task image_object_detection --agent pytorch --model_name mobilenet_ssd_v1_0 --architecture cpu --num_warmup 2 --dataset_name test --batch_size 1 --gpu_trace false 
```

## Image Semantic Segmentation 

```bash 
python run_image_semantic_segmentation.py --task image_semantic_segmentation --agent tensorflow --model_name deeplabv3_mobilenet_v2_dm_05_pascal_voc_train_aug --architecture cpu --num_warmup 2 --dataset_name test_cv2 --batch_size 1 --gpu_trace false 
```

## Image Enhancement 

```bash 
python run_image_enhancement.py --task image_enhancement --agent pytorch --model_name srgan --architecture cpu --num_warmup 2 --dataset_name test --batch_size 1 --gpu_trace false 
```

## Image Instance Segmentation 

```bash 
python run_image_instance_segmentation.py --task image_instance_segmentation --agent tensorflow --model_name mask_rcnn_inception_v2_coco --architecture cpu --num_warmup 2 --dataset_name test_cv2 --batch_size 1 --gpu_trace false 
```

## Image Instance Segmentation Raw 

```bash 
python run_image_instance_segmentation_raw.py --task image_instance_segmentation_raw --agent tensorflow --model_name mask_rcnn_inception_v2_coco_raw --architecture cpu --num_warmup 2 --dataset_name test_cv2 --batch_size 1 --gpu_trace false 
```

# References 

<a id="1">[1]</a> c3sr, “GitHub - c3sr/mlmodelscope: MLModelScope is an open source, extensible, and customizable platform to facilitate evaluation and measurement of ML models within AI pipelines.,” GitHub. 

<a id="2">[2]</a> c3sr, “GitHub - c3sr/go-pytorch,” GitHub, Oct. 25, 2021. https://github.com/c3sr/go-pytorch

<a id="3">[3]</a> “PyTorch,” PyTorch. https://www.pytorch.org

<a id="4">[4]</a> “OpenTelemetry,” OpenTelemetry. https://opentelemetry.io/ 

<a id="5">[5]</a> “Jaeger: open source, end-to-end distributed tracing,” Jaeger: open source, end-to-end distributed tracing, May 30, 2022. https://www.jaegertracing.io/ 
