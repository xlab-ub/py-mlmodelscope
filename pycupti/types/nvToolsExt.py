import ctypes 

nvtxRangeId_t = ctypes.c_uint64 

# /* Forward declaration of opaque domain registration structure #
class nvtxStringRegistration_st(ctypes.Structure): 
    pass 
nvtxStringRegistration = nvtxStringRegistration_st 

# \brief Registered String Handle Structure.
# \anchor REGISTERED_STRING_HANDLE_STRUCTURE
#
# This structure is opaque to the user and is used as a handle to reference
# a registered string.  This type is returned from tools when using the NVTX
# API to create a registered string.
#
#
nvtxStringHandle_t = ctypes.POINTER(nvtxStringRegistration)

class nvtxMessageValue_t(ctypes.Union): 
    _fields_ = [("ascii", ctypes.c_char_p), 
                ("unicode", ctypes.c_wchar_p), 
                ("registered", nvtxStringHandle_t) 
    ] 

#
# \brief Payload assigned to this event. \anchor PAYLOAD_FIELD
#
# A numerical value that can be used to annotate an event. The tool could
# use the payload data to reconstruct graphs and diagrams.
#
class payload_t(ctypes.Union): 
    _fields_ = [("ullValue", ctypes.c_uint64), 
                ("llValue", ctypes.c_int64), 
                ("dValue", ctypes.c_double), 
                # /* NVTX_VERSION_2 # 
                ("uiValue", ctypes.c_uint32), 
                ("iValue", ctypes.c_int32), 
                ("fValue", ctypes.c_float) 
    ] 

# \brief Event Attribute Structure.
# \anchor EVENT_ATTRIBUTE_STRUCTURE
#
# This structure is used to describe the attributes of an event. The layout of
# the structure is defined by a specific version of the tools extension
# library and can change between different versions of the Tools Extension
# library.
#
# \par Initializing the Attributes
#
# The caller should always perform the following three tasks when using
# attributes:
# <ul>
#    <li>Zero the structure
#    <li>Set the version field
#    <li>Set the size field
# </ul>
#
# Zeroing the structure sets all the event attributes types and values
# to the default value.
#
# The version and size field are used by the Tools Extension
# implementation to handle multiple versions of the attributes structure.
#
# It is recommended that the caller use one of the following to methods
# to initialize the event attributes structure:
#
# \par Method 1: Initializing nvtxEventAttributes for future compatibility
# \code
# nvtxEventAttributes_t eventAttrib = {0};
# eventAttrib.version = NVTX_VERSION;
# eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
# \endcode
#
# \par Method 2: Initializing nvtxEventAttributes for a specific version
# \code
# nvtxEventAttributes_t eventAttrib = {0};
# eventAttrib.version = 1;
# eventAttrib.size = (uint16_t)(sizeof(nvtxEventAttributes_v1));
# \endcode
#
# If the caller uses Method 1 it is critical that the entire binary
# layout of the structure be configured to 0 so that all fields
# are initialized to the default value.
#
# The caller should either use both NVTX_VERSION and
# NVTX_EVENT_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
# and a versioned type (Method 2).  Using a mix of the two methods
# will likely cause either source level incompatibility or binary
# incompatibility in the future.
#
# \par Settings Attribute Types and Values
#
#
# \par Example:
# \code
# // Initialize
# nvtxEventAttributes_t eventAttrib = {0};
# eventAttrib.version = NVTX_VERSION;
# eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
#
# // Configure the Attributes
# eventAttrib.colorType = NVTX_COLOR_ARGB;
# eventAttrib.color = 0xFF880000;
# eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
# eventAttrib.message.ascii = "Example";
# \endcode
#
# In the example the caller does not have to set the value of
# \ref ::nvtxEventAttributes_v2::category or
# \ref ::nvtxEventAttributes_v2::payload as these fields were set to
# the default value by {0}.
# \sa
# ::nvtxDomainMarkEx
# ::nvtxDomainRangeStartEx
# ::nvtxDomainRangePushEx
#
class nvtxEventAttributes_v2(ctypes.Structure): 
    _fields_ = [#
                # \brief Version flag of the structure.
                #
                # Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
                # supported in this header file. This can optionally be overridden to
                # another version of the tools extension library.
                #
                ("version", ctypes.c_uint16), 

                #
                # \brief Size of the structure.
                #
                # Needs to be set to the size in bytes of the event attribute
                # structure used to specify the event.
                #
                ("size", ctypes.c_uint16), 

                #
                # \brief ID of the category the event is assigned to.
                #
                # A category is a user-controlled ID that can be used to group
                # events.  The tool may use category IDs to improve filtering or
                # enable grouping of events in the same category. The functions
                # \ref ::nvtxNameCategoryA or \ref ::nvtxNameCategoryW can be used
                # to name a category.
                #
                # Default Value is 0
                #
                ("category", ctypes.c_uint32), 

                # \brief Color type specified in this attribute structure.
                #
                # Defines the color format of the attribute structure's \ref COLOR_FIELD
                # "color" field.
                #
                # Default Value is NVTX_COLOR_UNKNOWN
                #
                ("colorType", ctypes.c_int32),      # /* nvtxColorType_t # 

                # \brief Color assigned to this event. \anchor COLOR_FIELD
                #
                # The color that the tool should use to visualize the event.
                #
                ("color", ctypes.c_uint32), 

                #
                # \brief Payload type specified in this attribute structure.
                #
                # Defines the payload format of the attribute structure's \ref PAYLOAD_FIELD
                # "payload" field.
                #
                # Default Value is NVTX_PAYLOAD_UNKNOWN
                #
                ("payloadType", ctypes.c_int32),    # /* nvtxPayloadType_t # 

                ("reserved0", ctypes.c_int32), 

                ("payload", payload_t), 

                # \brief Message type specified in this attribute structure.
                #
                # Defines the message format of the attribute structure's \ref MESSAGE_FIELD
                # "message" field.
                #
                # Default Value is NVTX_MESSAGE_UNKNOWN
                #
                ("messageType", ctypes.c_int32),    # /* nvtxMessageType_t # 

                # \brief Message assigned to this attribute structure. \anchor MESSAGE_FIELD
                #
                # The text message that is attached to an event.
                #
                ("message", nvtxMessageValue_t) 
    ] 

nvtxEventAttributes_t = nvtxEventAttributes_v2 
