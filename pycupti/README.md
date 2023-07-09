# py-cupti

Python binding to NVIDIA CUPTI, the CUDA Performance Tool Interface.

## Pre-requsite Dynamic Library Installation 

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

## CUDA Profiling Tools Interface (CUPTI) 

The [CUDA Profiling Tools Interface (CUPTI)](https://docs.nvidia.com/cupti/Cupti/index.html) enables the creation of profiling and tracing tools that target CUDA applications. CUPTI provides four APIs: the Activity API, the Callback API, the Event API, and the Metric API.
1. The CUPTI Activity API allows you to asynchronously collect a trace of an application's CPU and GPU CUDA activity.
  - Activity Record. CPU and GPU activity is reported in C data structures called activity records.
  - Activity Buffer. An activity buffer is used to transfer one or more activity records from CUPTI to the client. An asynchronous buffering API is implemented by cuptiActivityRegisterCallbacks and cuptiActivityFlushAll.
2. The CUPTI Callback API allows you to register a callback into your own code. Your callback will be invoked when the application being profiled calls a CUDA runtime or driver function, or when certain events occur in the CUDA driver.
  - Callback Domain.
  - Callback ID. Each callback is given a unique ID within the corresponding callback domain so that you can identify it within your callback function.
  - Callback Function
    Your callback function must be of type CUpti_CallbackFunc. This function type has two arguments that specify the callback domain and ID so that you know why the callback is occurring. The type also has a cbdata argument that is used to pass data specific to the callback.
  - Subscriber. A subscriber is used to associate each of your callback functions with one or more CUDA API functions. There can be at most one subscriber initialized with cuptiSubscribe() at any time. Before initializing a new subscriber, the existing subscriber must be finalized with cuptiUnsubscribe().
3. The CUPTI Event API allows you to query, configure, start, stop, and read the event counters on a CUDA-enabled device.
4. The CUPTI Metric API allows you to collect application metrics calculated from one or more event values. 

## References 

<a id="1">[1]</a> “CUPTI :: CUPTI Documentation,” CUPTI :: CUPTI Documentation. https://docs.nvidia.com/cupti 

<a id="2">[2]</a> c3sr, “GitHub - c3sr/go-cupti: Use NVIDIA CUPTI from within GO,” GitHub, Sep. 18, 2021. https://github.com/c3sr/go-cupti 

<a id="3">[3]</a> sree314, “GitHub - sree314/pycupti: A thin Python wrapper around libcupti,” GitHub, Jul. 01, 2020. https://github.com/sree314/pycupti 

<a id="4">[4]</a> “ctypes — A foreign function library for Python,” Python documentation. https://docs.python.org/3/library/ctypes.html 

<a id="5">[5]</a> “CUDA Runtime API :: CUDA Toolkit Documentation,” CUDA Runtime API :: CUDA Toolkit Documentation. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html 
