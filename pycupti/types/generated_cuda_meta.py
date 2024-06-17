from .cuda import CUdeviceptr, CUfunction, CUstream 
from .driver_types import cudaMemcpyKind 
import ctypes 

# // *************************************************************************
# //      Definitions of structs to hold parameters for each function
# // *************************************************************************

class cudaMallocManaged_v6000_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.POINTER(ctypes.c_void_p)), 
                ("size", ctypes.c_size_t), 
                ("flags", ctypes.c_uint) 
    ] 

class cudaMalloc_v3020_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.POINTER(ctypes.c_void_p)), 
                ("size", ctypes.c_size_t)
    ] 

class cudaMallocHost_v3020_params(ctypes.Structure): 
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_void_p)), 
                ("size", ctypes.c_size_t)
    ] 

class cudaMallocPitch_v3020_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.POINTER(ctypes.c_void_p)), 
                ("pitch", ctypes.POINTER(ctypes.c_size_t)), 
                ("width", ctypes.c_size_t), 
                ("height", ctypes.c_size_t) 
    ] 

class cudaFree_v3020_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.c_void_p), 
    ] 

class cudaHostAlloc_v3020_params(ctypes.Structure): 
    _fields_ = [("pHost", ctypes.POINTER(ctypes.c_void_p)), 
                ("size", ctypes.c_size_t), 
                ("flags", ctypes.c_uint) 
    ] 

class cudaMemcpy_v3020_params(ctypes.Structure): 
    _fields_ = [("dst", ctypes.c_void_p), 
                ("src", ctypes.c_void_p), 
                ("count", ctypes.c_size_t), 
                ("kind", cudaMemcpyKind)
    ]

class cuMemcpyHtoD_v2_params(ctypes.Structure): 
    _fields_ = [("dstDevice", ctypes.c_void_p), 
                ("srcHost", ctypes.c_void_p), 
                ("ByteCount", ctypes.c_size_t) 
    ]

class cuLaunchKernel_params(ctypes.Structure): 
    _fields_ = [("f", CUfunction), 
                ("gridDimX", ctypes.c_uint), 
                ("gridDimY", ctypes.c_uint), 
                ("gridDimZ", ctypes.c_uint), 
                ("blockDimX", ctypes.c_uint), 
                ("blockDimY", ctypes.c_uint), 
                ("blockDimZ", ctypes.c_uint), 
                ("sharedMemBytes", ctypes.c_uint), 
                ("hStream", CUstream), 
                ("kernelParams", ctypes.c_void_p), 
                ("extra", ctypes.c_void_p) 
    ]

class cuLaunchCooperativeKernel_params(ctypes.Structure):
    _fields_ = [("f", CUfunction), 
                ("gridDimX", ctypes.c_uint), 
                ("gridDimY", ctypes.c_uint), 
                ("gridDimZ", ctypes.c_uint), 
                ("blockDimX", ctypes.c_uint), 
                ("blockDimY", ctypes.c_uint), 
                ("blockDimZ", ctypes.c_uint), 
                ("sharedMemBytes", ctypes.c_uint), 
                ("hStream", CUstream), 
                ("kernelParams", ctypes.c_void_p)
    ]