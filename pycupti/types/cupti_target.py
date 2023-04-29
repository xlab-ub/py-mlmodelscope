import ctypes 

class CUpti_Device_GetChipName_Params(ctypes.Structure): 
    _fields_ = [("structSize", ctypes.c_size_t),    # //!< [in]
                ("pPriv", ctypes.c_void_p),         # //!< [in] assign to NULL

                ("deviceIndex", ctypes.c_size_t),   # //!< [in]
                ("pChipName", ctypes.c_char_p)      # //!< [out]
    ]