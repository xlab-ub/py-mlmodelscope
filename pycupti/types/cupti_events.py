import ctypes 
from aenum import IntEnum, export 

#
# \brief ID for an event.
#
# An event represents a countable activity, action, or occurrence on
# the device.
#
CUpti_EventID = ctypes.c_uint32 

#
# \brief ID for an event domain.
#
# ID for an event domain. An event domain represents a group of
# related events. A device may have multiple instances of a domain,
# indicating that the device can simultaneously record multiple
# instances of each event within that domain.
#
CUpti_EventDomainID = ctypes.c_uint32 

#
# \brief A group of events.
#
# An event group is a collection of events that are managed
# together. All events in an event group must belong to the same
# domain.
#
CUpti_EventGroup = ctypes.c_void_p 

#
# \brief Device attributes.
#
# CUPTI device attributes. These attributes can be read using \ref
# cuptiDeviceGetAttribute.
#
CUpti_DeviceAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_DeviceAttribute_(IntEnum): 
  #
  # Number of event IDs for a device. Value is a uint32_t.
  #
  CUPTI_DEVICE_ATTR_MAX_EVENT_ID                            = 1
  #
  # Number of event domain IDs for a device. Value is a uint32_t.
  #
  CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID                     = 2
  #
  # Get global memory bandwidth in Kbytes/sec. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH                 = 3
  #
  # Get theoretical maximum number of instructions per cycle. Value
  # is a uint32_t.
  #
  CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE                   = 4
  #
  # Get theoretical maximum number of single precision instructions
  # that can be executed per second. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION = 5
  #
  # Get number of frame buffers for device.  Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS                       = 6
  #
  # Get PCIE link rate in Mega bits/sec for device. Return 0 if bus-type
  # is non-PCIE. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_PCIE_LINK_RATE                          = 7
  #
  # Get PCIE link width for device. Return 0 if bus-type
  # is non-PCIE. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH                         = 8
  #
  # Get PCIE generation for device. Return 0 if bus-type
  # is non-PCIE. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_PCIE_GEN                                = 9
  #
  # Get the class for the device. Value is a
  # CUpti_DeviceAttributeDeviceClass.
  #
  CUPTI_DEVICE_ATTR_DEVICE_CLASS                            = 10
  #
  # Get the peak single precision flop per cycle. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE                       = 11
  #
  # Get the peak double precision flop per cycle. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE                       = 12
  #
  # Get number of L2 units. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_MAX_L2_UNITS                           = 13
  #
  # Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_SHARED
  # preference. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED = 14
  #
  # Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_L1
  # preference. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 = 15
  #
  # Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_EQUAL
  # preference. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL = 16
  #
  # Get the peak half precision flop per cycle. Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE                       = 17
  #
  # Check if Nvlink is connected to device. Returns 1, if at least one
  # Nvlink is connected to the device, returns 0 otherwise.
  # Value is a uint32_t.
  #
  CUPTI_DEVICE_ATTR_NVLINK_PRESENT                          = 18
  #
  # Check if Nvlink is present between GPU and CPU. Returns Bandwidth,
  # in Bytes/sec, if Nvlink is present, returns 0 otherwise.
  # Value is a uint64_t.
  #
  CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW                       = 19
  #
  # Check if NVSwitch is present in the underlying topology.
  # Returns 1, if present, returns 0 otherwise.
  # Value is a uint32_t.
  #
  CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT                        = 20
  CUPTI_DEVICE_ATTR_FORCE_INT                               = 0x7fffffff

#
# \brief Event domain attributes.
#
# Event domain attributes. Except where noted, all the attributes can
# be read using either \ref cuptiDeviceGetEventDomainAttribute or
# \ref cuptiEventDomainGetAttribute.
#
CUpti_EventDomainAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_EventDomainAttribute_(IntEnum): 
  #
  # Event domain name. Value is a null terminated const c-string.
  #
  CUPTI_EVENT_DOMAIN_ATTR_NAME                 = 0
  #
  # Number of instances of the domain for which event counts will be
  # collected.  The domain may have additional instances that cannot
  # be profiled (see CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT).
  # Can be read only with \ref
  # cuptiDeviceGetEventDomainAttribute. Value is a uint32_t.
  #
  CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT       = 1
  #
  # Total number of instances of the domain, including instances that
  # cannot be profiled.  Use CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT
  # to get the number of instances that can be profiled. Can be read
  # only with \ref cuptiDeviceGetEventDomainAttribute. Value is a
  # uint32_t.
  #
  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT = 3
  #
  # Collection method used for events contained in the event domain.
  # Value is a \ref CUpti_EventCollectionMethod.
  #
  CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD    = 4

  CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT      = 0x7fffffff

#
# \brief Event group attributes.
#
# Event group attributes. These attributes can be read using \ref
# cuptiEventGroupGetAttribute. Attributes marked [rw] can also be
# written using \ref cuptiEventGroupSetAttribute.
#
CUpti_EventGroupAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_EventGroupAttribute_(IntEnum): 
  #
  # The domain to which the event group is bound. This attribute is
  # set when the first event is added to the group.  Value is a
  # CUpti_EventDomainID.
  #
  CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID              = 0
  #
  # [rw] Profile all the instances of the domain for this
  # eventgroup. This feature can be used to get load balancing
  # across all instances of a domain. Value is an integer.
  #
  CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES = 1
  #
  # [rw] Reserved for user data.
  #
  CUPTI_EVENT_GROUP_ATTR_USER_DATA                    = 2
  #
  # Number of events in the group. Value is a uint32_t.
  #
  CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS                   = 3
  #
  # Enumerates events in the group. Value is a pointer to buffer of
  # size sizeof(CUpti_EventID)# num_of_events in the eventgroup.
  # num_of_events can be queried using
  # CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS.
  #
  CUPTI_EVENT_GROUP_ATTR_EVENTS                       = 4
  #
  # Number of instances of the domain bound to this event group that
  # will be counted.  Value is a uint32_t.
  #
  CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT               = 5
  #
  # Event group scope can be set to CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
  # CUPTI_EVENT_PROFILING_SCOPE_CONTEXT for an eventGroup, before
  # adding any event.
  # Sets the scope of eventgroup as CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
  # CUPTI_EVENT_PROFILING_SCOPE_CONTEXT when the scope of the events
  # that will be added is CUPTI_EVENT_PROFILING_SCOPE_BOTH.
  # If profiling scope of event is either
  # CUPTI_EVENT_PROFILING_SCOPE_DEVICE or CUPTI_EVENT_PROFILING_SCOPE_CONTEXT
  # then setting this attribute will not affect the default scope.
  # It is not allowed to add events of different scope to same eventgroup.
  # Value is a uint32_t.
  #
  CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE               = 6
  CUPTI_EVENT_GROUP_ATTR_FORCE_INT                     = 0x7fffffff

#
# \brief Event attributes.
#
# Event attributes. These attributes can be read using \ref
# cuptiEventGetAttribute.
#
CUpti_EventAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_EventAttribute_(IntEnum): 
  #
  # Event name. Value is a null terminated const c-string.
  #
  CUPTI_EVENT_ATTR_NAME              = 0
  #
  # Short description of event. Value is a null terminated const
  # c-string.
  #
  CUPTI_EVENT_ATTR_SHORT_DESCRIPTION = 1
  #
  # Long description of event. Value is a null terminated const
  # c-string.
  #
  CUPTI_EVENT_ATTR_LONG_DESCRIPTION  = 2
  #
  # Category of event. Value is CUpti_EventCategory.
  #
  CUPTI_EVENT_ATTR_CATEGORY          = 3
  #
  # Profiling scope of the events. It can be either device or context or both.
  # Value is a \ref CUpti_EventProfilingScope.
  #
  CUPTI_EVENT_ATTR_PROFILING_SCOPE   = 5

  CUPTI_EVENT_ATTR_FORCE_INT         = 0x7fffffff

#
# \brief A set of event groups.
#
# A set of event groups. When returned by \ref
# cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
# a set indicates that event groups that can be enabled at the same
# time (i.e. all the events in the set can be collected
# simultaneously).
#
class CUpti_EventGroupSet(ctypes.Structure): 
  _fields_ = [#
              # The number of event groups in the set.
              #
              ("numEventGroups", ctypes.c_uint32), 
              #
              # An array of \p numEventGroups event groups.
              #
              ("eventGroups", ctypes.POINTER(CUpti_EventGroup)) 
  ]

#
# \brief A set of event group sets.
#
# A set of event group sets. When returned by \ref
# cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
# a CUpti_EventGroupSets indicates the number of passes required to
# collect all the events, and the event groups that should be
# collected during each pass.
#
class CUpti_EventGroupSets(ctypes.Structure): 
  _fields_ = [#
              # Number of event group sets.
              #
              ("numSets", ctypes.c_uint32), 
              #
              # An array of \p numSets event group sets.
              #
              ("sets", ctypes.POINTER(CUpti_EventGroupSet)) 
  ]
  