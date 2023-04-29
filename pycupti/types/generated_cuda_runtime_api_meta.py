from .driver_types import cudaIpcEventHandle_t, cudaEvent_t 
import ctypes 

# // *************************************************************************
# //      Definitions of structs to hold parameters for each function
# // *************************************************************************

class cudaIpcGetEventHandle_v4010_params(ctypes.Structure): 
    _fields_ = [("handle", ctypes.POINTER(cudaIpcEventHandle_t)), 
                ("event", cudaEvent_t) 
    ] 

class cudaIpcOpenEventHandle_v4010_params(ctypes.Structure): 
    _fields_ = [("event", ctypes.POINTER(cudaEvent_t)), 
                ("handle", cudaIpcEventHandle_t) 
    ] 

class cudaIpcGetMemHandle_v4010_params(ctypes.Structure): 
    _fields_ = [("handle", ctypes.POINTER(cudaIpcEventHandle_t)), 
                ("devPtr", ctypes.c_void_p) 
    ] 

class cudaIpcOpenMemHandle_v4010_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.POINTER(ctypes.c_void_p)), 
                ("handle", cudaIpcEventHandle_t), 
                ("flags", ctypes.c_uint) 
    ] 

class cudaIpcCloseMemHandle_v4010_params(ctypes.Structure): 
    _fields_ = [("devPtr", ctypes.c_void_p) 
    ] 