from .nvToolsExt import nvtxEventAttributes_t, nvtxRangeId_t 
import ctypes 

# // *************************************************************************
# //      Definitions of structs to hold parameters for each function
# // *************************************************************************

class nvtxRangeStartEx_params(ctypes.Structure): 
    _fields_ = [("eventAttrib", ctypes.POINTER(nvtxEventAttributes_t)) 
    ] 

class nvtxRangeStartA_params(ctypes.Structure): 
    _fields_ = [("message", ctypes.c_char_p) 
    ] 

class nvtxRangeStartW_params(ctypes.Structure): 
    _fields_ = [("message", ctypes.c_wchar_p) 
    ] 
    
class nvtxRangeEnd_params(ctypes.Structure): 
    _fields_ = [("id", nvtxRangeId_t) 
    ] 

class nvtxRangePushEx_params(ctypes.Structure): 
    _fields_ = [("eventAttrib", ctypes.POINTER(nvtxEventAttributes_t)) 
    ] 

class nvtxRangePushA_params(ctypes.Structure): 
    _fields_ = [("message", ctypes.c_char_p) 
    ] 

class nvtxRangePushW_params(ctypes.Structure): 
    _fields_ = [("message", ctypes.c_wchar_p) 
    ] 