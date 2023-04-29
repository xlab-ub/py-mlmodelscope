from .cuda import * 
import ctypes 
from aenum import IntEnum, export 

#
# CUDA memory copy types
#
cudaMemcpyKind = ctypes.c_int 
@export(globals()) 
class cudaMemcpyKind_(IntEnum): 
    cudaMemcpyHostToHost          =   0     # /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1     # /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2     # /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3     # /**< Device -> Device */
    cudaMemcpyDefault             =   4     # /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */

#
# CUDA IPC Handle Size
#
CUDA_IPC_HANDLE_SIZE = 64 

#
# CUDA IPC event handle
#
class cudaIpcEventHandle_t(ctypes.Structure): 
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_SIZE) 
    ] 

#
# CUDA event types
#
cudaEvent_t = ctypes.POINTER(CUevent_st) 