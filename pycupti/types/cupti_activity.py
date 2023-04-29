from .cuda import CUaccessPolicyWindow 
from .cupti_callbacks import CUpti_CallbackId 
import ctypes 
from aenum import IntEnum, export 

#
# \brief The kinds of activity records.
#
# Each activity record kind represents information about a GPU or an
# activity occurring on a CPU or GPU. Each kind is associated with a
# activity record structure that holds the information associated
# with the kind.
# \see CUpti_Activity
# \see CUpti_ActivityAPI
# \see CUpti_ActivityContext
# \see CUpti_ActivityDevice
# \see CUpti_ActivityDevice2
# \see CUpti_ActivityDevice3
# \see CUpti_ActivityDevice4
# \see CUpti_ActivityDeviceAttribute
# \see CUpti_ActivityEvent
# \see CUpti_ActivityEventInstance
# \see CUpti_ActivityKernel
# \see CUpti_ActivityKernel2
# \see CUpti_ActivityKernel3
# \see CUpti_ActivityKernel4
# \see CUpti_ActivityKernel5
# \see CUpti_ActivityKernel6
# \see CUpti_ActivityKernel7
# \see CUpti_ActivityCdpKernel
# \see CUpti_ActivityPreemption
# \see CUpti_ActivityMemcpy
# \see CUpti_ActivityMemcpy3
# \see CUpti_ActivityMemcpy4
# \see CUpti_ActivityMemcpy5
# \see CUpti_ActivityMemcpyPtoP
# \see CUpti_ActivityMemcpyPtoP2
# \see CUpti_ActivityMemcpyPtoP3
# \see CUpti_ActivityMemcpyPtoP4
# \see CUpti_ActivityMemset
# \see CUpti_ActivityMemset2
# \see CUpti_ActivityMemset3
# \see CUpti_ActivityMemset4
# \see CUpti_ActivityMetric
# \see CUpti_ActivityMetricInstance
# \see CUpti_ActivityName
# \see CUpti_ActivityMarker
# \see CUpti_ActivityMarker2
# \see CUpti_ActivityMarkerData
# \see CUpti_ActivitySourceLocator
# \see CUpti_ActivityGlobalAccess
# \see CUpti_ActivityGlobalAccess2
# \see CUpti_ActivityGlobalAccess3
# \see CUpti_ActivityBranch
# \see CUpti_ActivityBranch2
# \see CUpti_ActivityOverhead
# \see CUpti_ActivityEnvironment
# \see CUpti_ActivityInstructionExecution
# \see CUpti_ActivityUnifiedMemoryCounter
# \see CUpti_ActivityFunction
# \see CUpti_ActivityModule
# \see CUpti_ActivitySharedAccess
# \see CUpti_ActivityPCSampling
# \see CUpti_ActivityPCSampling2
# \see CUpti_ActivityPCSampling3
# \see CUpti_ActivityPCSamplingRecordInfo
# \see CUpti_ActivityCudaEvent
# \see CUpti_ActivityStream
# \see CUpti_ActivitySynchronization
# \see CUpti_ActivityInstructionCorrelation
# \see CUpti_ActivityExternalCorrelation
# \see CUpti_ActivityUnifiedMemoryCounter2
# \see CUpti_ActivityOpenAccData
# \see CUpti_ActivityOpenAccLaunch
# \see CUpti_ActivityOpenAccOther
# \see CUpti_ActivityOpenMp
# \see CUpti_ActivityNvLink
# \see CUpti_ActivityNvLink2
# \see CUpti_ActivityNvLink3
# \see CUpti_ActivityNvLink4
# \see CUpti_ActivityMemory
# \see CUpti_ActivityPcie
#
CUpti_ActivityKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityKind_(IntEnum): 
  #
  # The activity record is invalid.
  #
  CUPTI_ACTIVITY_KIND_INVALID  = 0
  #
  # A host<->host, host<->device, or device<->device memory copy. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityMemcpy5.
  #
  CUPTI_ACTIVITY_KIND_MEMCPY   = 1
  #
  # A memory set executing on the GPU. The corresponding activity
  # record structure is \ref CUpti_ActivityMemset4.
  #
  CUPTI_ACTIVITY_KIND_MEMSET   = 2
  #
  # A kernel executing on the GPU. This activity kind may significantly change
  # the overall performance characteristics of the application because all
  # kernel executions are serialized on the GPU. Other activity kind for kernel
  # CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL doesn't break kernel concurrency.
  # The corresponding activity record structure is \ref CUpti_ActivityKernel7.
  #
  CUPTI_ACTIVITY_KIND_KERNEL   = 3
  #
  # A CUDA driver API function execution. The corresponding activity
  # record structure is \ref CUpti_ActivityAPI.
  #
  CUPTI_ACTIVITY_KIND_DRIVER   = 4
  #
  # A CUDA runtime API function execution. The corresponding activity
  # record structure is \ref CUpti_ActivityAPI.
  #
  CUPTI_ACTIVITY_KIND_RUNTIME  = 5
  #
  # An event value. The corresponding activity record structure is
  # \ref CUpti_ActivityEvent.
  #
  CUPTI_ACTIVITY_KIND_EVENT    = 6
  #
  # A metric value. The corresponding activity record structure is
  # \ref CUpti_ActivityMetric.
  #
  CUPTI_ACTIVITY_KIND_METRIC   = 7
  #
  # Information about a device. The corresponding activity record
  # structure is \ref CUpti_ActivityDevice4.
  #
  CUPTI_ACTIVITY_KIND_DEVICE   = 8
  #
  # Information about a context. The corresponding activity record
  # structure is \ref CUpti_ActivityContext.
  #
  CUPTI_ACTIVITY_KIND_CONTEXT  = 9
  #
  # A kernel executing on the GPU. This activity kind doesn't break
  # kernel concurrency. The corresponding activity record structure
  # is \ref CUpti_ActivityKernel7.
  #
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
  #
  # Resource naming done via NVTX APIs for thread, device, context, etc.
  # The corresponding activity record structure is \ref CUpti_ActivityName.
  #
  CUPTI_ACTIVITY_KIND_NAME     = 11
  #
  # Instantaneous, start, or end NVTX marker. The corresponding activity
  # record structure is \ref CUpti_ActivityMarker2.
  #
  CUPTI_ACTIVITY_KIND_MARKER = 12
  #
  # Extended, optional, data about a marker. The corresponding
  # activity record structure is \ref CUpti_ActivityMarkerData.
  #
  CUPTI_ACTIVITY_KIND_MARKER_DATA = 13
  #
  # Source information about source level result. The corresponding
  # activity record structure is \ref CUpti_ActivitySourceLocator.
  #
  CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR = 14
  #
  # Results for source-level global acccess. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityGlobalAccess3.
  #
  CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS = 15
  #
  # Results for source-level branch. The corresponding
  # activity record structure is \ref CUpti_ActivityBranch2.
  #
  CUPTI_ACTIVITY_KIND_BRANCH = 16
  #
  # Overhead activity records. The
  # corresponding activity record structure is
  # \ref CUpti_ActivityOverhead.
  #
  CUPTI_ACTIVITY_KIND_OVERHEAD = 17
  #
  # A CDP (CUDA Dynamic Parallel) kernel executing on the GPU. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityCdpKernel.  This activity can not be directly
  # enabled or disabled. It is enabled and disabled through
  # concurrent kernel activity i.e. _CONCURRENT_KERNEL.
  #
  CUPTI_ACTIVITY_KIND_CDP_KERNEL = 18
  #
  # Preemption activity record indicating a preemption of a CDP (CUDA
  # Dynamic Parallel) kernel executing on the GPU. The corresponding
  # activity record structure is \ref CUpti_ActivityPreemption.
  #
  CUPTI_ACTIVITY_KIND_PREEMPTION = 19
  #
  # Environment activity records indicating power, clock, thermal,
  # etc. levels of the GPU. The corresponding activity record
  # structure is \ref CUpti_ActivityEnvironment.
  #
  CUPTI_ACTIVITY_KIND_ENVIRONMENT = 20
  #
  # An event value associated with a specific event domain
  # instance. The corresponding activity record structure is \ref
  # CUpti_ActivityEventInstance.
  #
  CUPTI_ACTIVITY_KIND_EVENT_INSTANCE = 21
  #
  # A peer to peer memory copy. The corresponding activity record
  # structure is \ref CUpti_ActivityMemcpyPtoP4.
  #
  CUPTI_ACTIVITY_KIND_MEMCPY2 = 22
  #
  # A metric value associated with a specific metric domain
  # instance. The corresponding activity record structure is \ref
  # CUpti_ActivityMetricInstance.
  #
  CUPTI_ACTIVITY_KIND_METRIC_INSTANCE = 23
  #
  # Results for source-level instruction execution.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstructionExecution.
  #
  CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION = 24
  #
  # Unified Memory counter record. The corresponding activity
  # record structure is \ref CUpti_ActivityUnifiedMemoryCounter2.
  #
  CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25
  #
  # Device global/function record. The corresponding activity
  # record structure is \ref CUpti_ActivityFunction.
  #
  CUPTI_ACTIVITY_KIND_FUNCTION = 26
  #
  # CUDA Module record. The corresponding activity
  # record structure is \ref CUpti_ActivityModule.
  #
  CUPTI_ACTIVITY_KIND_MODULE = 27
  #
  # A device attribute value. The corresponding activity record
  # structure is \ref CUpti_ActivityDeviceAttribute.
  #
  CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE   = 28
  #
  # Results for source-level shared acccess. The
  # corresponding activity record structure is \ref
  # CUpti_ActivitySharedAccess.
  #
  CUPTI_ACTIVITY_KIND_SHARED_ACCESS = 29
  #
  # Enable PC sampling for kernels. This will serialize
  # kernels. The corresponding activity record structure
  # is \ref CUpti_ActivityPCSampling3.
  #
  CUPTI_ACTIVITY_KIND_PC_SAMPLING = 30
  #
  # Summary information about PC sampling records. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityPCSamplingRecordInfo.
  #
  CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO = 31
  #
  # SASS/Source line-by-line correlation record.
  # This will generate sass/source correlation for functions that have source
  # level analysis or pc sampling results. The records will be generated only
  # when either of source level analysis or pc sampling activity is enabled.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstructionCorrelation.
  #
  CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION = 32
  #
  # OpenACC data events.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityOpenAccData.
  #
  CUPTI_ACTIVITY_KIND_OPENACC_DATA = 33
  #
  # OpenACC launch events.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityOpenAccLaunch.
  #
  CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH = 34
  #
  # OpenACC other events.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityOpenAccOther.
  #
  CUPTI_ACTIVITY_KIND_OPENACC_OTHER = 35
  #
  # Information about a CUDA event. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityCudaEvent.
  #
  CUPTI_ACTIVITY_KIND_CUDA_EVENT = 36
  #
  # Information about a CUDA stream. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityStream.
  #
  CUPTI_ACTIVITY_KIND_STREAM = 37
  #
  # Records for synchronization management. The
  # corresponding activity record structure is \ref
  # CUpti_ActivitySynchronization.
  #
  CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38
  #
  # Records for correlation of different programming APIs. The
  # corresponding activity record structure is \ref
  # CUpti_ActivityExternalCorrelation.
  #
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39
  #
  # NVLink information.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityNvLink4.
  #
  CUPTI_ACTIVITY_KIND_NVLINK = 40
  #
  # Instantaneous Event information.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstantaneousEvent.
  #
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT = 41
  #
  # Instantaneous Event information for a specific event
  # domain instance.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstantaneousEventInstance
  #
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE = 42
  #
  # Instantaneous Metric information
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstantaneousMetric.
  #
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC = 43
  #
  # Instantaneous Metric information for a specific metric
  # domain instance.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityInstantaneousMetricInstance.
  #
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44
  #
  # Memory activity tracking allocation and freeing of the memory
  # The corresponding activity record structure is \ref
  # CUpti_ActivityMemory.
  #
  CUPTI_ACTIVITY_KIND_MEMORY = 45
  #
  # PCI devices information used for PCI topology.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityPcie.
  #
  CUPTI_ACTIVITY_KIND_PCIE = 46
  #
  # OpenMP parallel events.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityOpenMp.
  #
  CUPTI_ACTIVITY_KIND_OPENMP = 47
  #
  # A CUDA driver kernel launch occurring outside of any
  # public API function execution.  Tools can handle these
  # like records for driver API launch functions, although
  # the cbid field is not used here.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityAPI.
  #
  CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API = 48
  #
  # Memory activity tracking allocation and freeing of the memory
  # The corresponding activity record structure is \ref
  # CUpti_ActivityMemory3.
  #
  CUPTI_ACTIVITY_KIND_MEMORY2 = 49

  #
  # Memory pool activity tracking creation, destruction and
  # triming of the memory pool.
  # The corresponding activity record structure is \ref
  # CUpti_ActivityMemoryPool2.
  #
  CUPTI_ACTIVITY_KIND_MEMORY_POOL = 50

  #
  # The corresponding activity record structure is \ref CUpti_ActivityGraphTrace.
  #
  CUPTI_ACTIVITY_KIND_GRAPH_TRACE = 51



  CUPTI_ACTIVITY_KIND_COUNT = 52

  CUPTI_ACTIVITY_KIND_FORCE_INT     = 0x7fffffff


#
# \brief The kinds of activity objects.
# \see CUpti_ActivityObjectKindId
#
CUpti_ActivityObjectKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityObjectKind_(IntEnum): 
  #
  # The object kind is not known.
  #
  CUPTI_ACTIVITY_OBJECT_UNKNOWN  = 0
  #
  # A process.
  #
  CUPTI_ACTIVITY_OBJECT_PROCESS  = 1
  #
  # A thread.
  #
  CUPTI_ACTIVITY_OBJECT_THREAD   = 2
  #
  # A device.
  #
  CUPTI_ACTIVITY_OBJECT_DEVICE   = 3
  #
  # A context.
  #
  CUPTI_ACTIVITY_OBJECT_CONTEXT  = 4
  #
  # A stream.
  #
  CUPTI_ACTIVITY_OBJECT_STREAM   = 5

  CUPTI_ACTIVITY_OBJECT_FORCE_INT = 0x7fffffff

#
# \brief Identifiers for object kinds as specified by
# CUpti_ActivityObjectKind.
# \see CUpti_ActivityObjectKind
#
class CUpti_ActivityObjectKindId_pt(ctypes.Structure): 
    _fields_ = [("processId", ctypes.c_uint32), 
                ("threadId", ctypes.c_uint32) 
    ]
class CUpti_ActivityObjectKindId_dcs(ctypes.Structure): 
    _fields_ = [("deviceId", ctypes.c_uint32), 
                ("contextId", ctypes.c_uint32), 
                ("streamId", ctypes.c_uint32) 
    ]
class CUpti_ActivityObjectKindId(ctypes.Union): 
    _fields_ = [#
                # A process object requires that we identify the process ID. A
                # thread object requires that we identify both the process and
                # thread ID.
                #
                ("pt", CUpti_ActivityObjectKindId_pt), 
                #
                # A device object requires that we identify the device ID. A
                # context object requires that we identify both the device and
                # context ID. A stream object requires that we identify device,
                # context, and stream ID.
                #
                ("dcs", CUpti_ActivityObjectKindId_dcs) 
    ] 

#
# \brief The kinds of activity overhead.
#
CUpti_ActivityOverheadKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityOverheadKind_(IntEnum): 
  #
  # The overhead kind is not known.
  #
  CUPTI_ACTIVITY_OVERHEAD_UNKNOWN               = 0
  #
  # Compiler(JIT) overhead.
  #
  CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER       = 1
  #
  # Activity buffer flush overhead.
  #
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH    = 1<<16
  #
  # CUPTI instrumentation overhead.
  #
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION = 2<<16
  #
  # CUPTI resource creation and destruction overhead.
  #
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE        = 3<<16
  CUPTI_ACTIVITY_OVERHEAD_FORCE_INT             = 0x7fffffff

#
# \brief The kind of a compute API.
#
CUpti_ActivityComputeApiKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityComputeApiKind_(IntEnum):
  #
  # The compute API is not known.
  #
  CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN    = 0
  #
  # The compute APIs are for CUDA.
  #
  CUPTI_ACTIVITY_COMPUTE_API_CUDA       = 1
  #
  # The compute APIs are for CUDA running
  # in MPS (Multi-Process Service) environment.
  #
  CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS   = 2

  CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT  = 0x7fffffff

#
# \brief Flags associated with activity records.
#
# Activity record flags. Flags can be combined by bitwise OR to
# associated multiple flags with an activity record. Each flag is
# specific to a certain activity kind, as noted below.
#
CUpti_ActivityFlag = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityFlag_(IntEnum): 
  #
  # Indicates the activity record has no flags.
  #
  CUPTI_ACTIVITY_FLAG_NONE          = 0

  #
  # Indicates the activity represents a device that supports
  # concurrent kernel execution. Valid for
  # CUPTI_ACTIVITY_KIND_DEVICE.
  #
  CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS  = 1 << 0

  #
  # Indicates if the activity represents a CUdevice_attribute value
  # or a CUpti_DeviceAttribute value. Valid for
  # CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
  #
  CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE  = 1 << 0

  #
  # Indicates the activity represents an asynchronous memcpy
  # operation. Valid for CUPTI_ACTIVITY_KIND_MEMCPY.
  #
  CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC  = 1 << 0

  #
  # Indicates the activity represents an instantaneous marker. Valid
  # for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS  = 1 << 0

  #
  # Indicates the activity represents a region start marker. Valid
  # for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_START  = 1 << 1

  #
  # Indicates the activity represents a region end marker. Valid for
  # CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_END  = 1 << 2

  #
  # Indicates the activity represents an attempt to acquire a user
  # defined synchronization object.
  # Valid for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE = 1 << 3

  #
  # Indicates the activity represents success in acquiring the
  # user defined synchronization object.
  # Valid for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS = 1 << 4

  #
  # Indicates the activity represents failure in acquiring the
  # user defined synchronization object.
  # Valid for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED = 1 << 5

  #
  # Indicates the activity represents releasing a reservation on
  # user defined synchronization object.
  # Valid for CUPTI_ACTIVITY_KIND_MARKER.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE = 1 << 6

  #
  # Indicates the activity represents a marker that does not specify
  # a color. Valid for CUPTI_ACTIVITY_KIND_MARKER_DATA.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE  = 1 << 0

  #
  # Indicates the activity represents a marker that specifies a color
  # in alpha-red-green-blue format. Valid for
  # CUPTI_ACTIVITY_KIND_MARKER_DATA.
  #
  CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB  = 1 << 1

  #
  # The number of bytes requested by each thread
  # Valid for CUpti_ActivityGlobalAccess3.
  #
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK  = 0xFF << 0
  #
  # If bit in this flag is set, the access was load, else it is a
  # store access. Valid for CUpti_ActivityGlobalAccess3.
  #
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD       = 1 << 8
  #
  # If this bit in flag is set, the load access was cached else it is
  # uncached. Valid for CUpti_ActivityGlobalAccess3.
  #
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED     = 1 << 9
  #
  # If this bit in flag is set, the metric value overflowed. Valid
  # for CUpti_ActivityMetric and CUpti_ActivityMetricInstance.
  #
  CUPTI_ACTIVITY_FLAG_METRIC_OVERFLOWED     = 1 << 0
  #
  # If this bit in flag is set, the metric value couldn't be
  # calculated. This occurs when a value(s) required to calculate the
  # metric is missing.  Valid for CUpti_ActivityMetric and
  # CUpti_ActivityMetricInstance.
  #
  CUPTI_ACTIVITY_FLAG_METRIC_VALUE_INVALID  = 1 << 1
    #
  # If this bit in flag is set, the source level metric value couldn't be
  # calculated. This occurs when a value(s) required to calculate the
  # source level metric cannot be evaluated.
  # Valid for CUpti_ActivityInstructionExecution.
  #
  CUPTI_ACTIVITY_FLAG_INSTRUCTION_VALUE_INVALID  = 1 << 0
  #
  # The mask for the instruction class, \ref CUpti_ActivityInstructionClass
  # Valid for CUpti_ActivityInstructionExecution and
  # CUpti_ActivityInstructionCorrelation
  #
  CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK    = 0xFF << 1
  #
  # When calling cuptiActivityFlushAll, this flag
  # can be set to force CUPTI to flush all records in the buffer, whether
  # finished or not
  #
  CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1 << 0

  #
  # The number of bytes requested by each thread
  # Valid for CUpti_ActivitySharedAccess.
  #
  CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_SIZE_MASK  = 0xFF << 0
  #
  # If bit in this flag is set, the access was load, else it is a
  # store access.  Valid for CUpti_ActivitySharedAccess.
  #
  CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_LOAD       = 1 << 8

  #
  # Indicates the activity represents an asynchronous memset
  # operation. Valid for CUPTI_ACTIVITY_KIND_MEMSET.
  #
  CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC  = 1 << 0

  #
  # Indicates the activity represents thrashing in CPU.
  # Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING in
  # CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
  #
  CUPTI_ACTIVITY_FLAG_THRASHING_IN_CPU = 1 << 0

 #
  # Indicates the activity represents page throttling in CPU.
  # Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING in
  # CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
  #
  CUPTI_ACTIVITY_FLAG_THROTTLING_IN_CPU = 1 << 0

  CUPTI_ACTIVITY_FLAG_FORCE_INT = 0x7fffffff

#
# \brief The stall reason for PC sampling activity.
#
CUpti_ActivityPCSamplingStallReason = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityPCSamplingStallReason_(IntEnum): 
  #
  # Invalid reason
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID      = 0
   #
  # No stall, instruction is selected for issue
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE         = 1
  #
  # Warp is blocked because next instruction is not yet available,
  # because of instruction cache miss, or because of branching effects
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH   = 2
  #
  # Instruction is waiting on an arithmatic dependency
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY   = 3
  #
  # Warp is blocked because it is waiting for a memory access to complete.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY   = 4
  #
  # Texture sub-system is fully utilized or has too many outstanding requests.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE   = 5
  #
  # Warp is blocked as it is waiting at __syncthreads() or at memory barrier.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC   = 6
  #
  # Warp is blocked waiting for __constant__ memory and immediate memory access to complete.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY   = 7
  #
  # Compute operation cannot be performed due to the required resources not
  # being available.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY   = 8
  #
  # Warp is blocked because there are too many pending memory operations.
  # In Kepler architecture it often indicates high number of memory replays.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE   = 9
  #
  # Warp was ready to issue, but some other warp issued instead.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED   = 10
  #
  # Miscellaneous reasons
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER   = 11
  #
  # Sleeping.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING   = 12
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT  = 0x7fffffff

#
# \brief Sampling period for PC sampling method
#
# Sampling period can be set using \ref cuptiActivityConfigurePCSampling
#
CUpti_ActivityPCSamplingPeriod = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityPCSamplingPeriod_(IntEnum): 
  #
  # The PC sampling period is not set.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID = 0
  #
  # Minimum sampling period available on the device.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN = 1
  #
  # Sampling period in lower range.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW = 2
  #
  # Medium sampling period.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID = 3
  #
  # Sampling period in higher range.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH = 4
  #
  # Maximum sampling period available on the device.
  #
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX = 5
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT = 0x7fffffff

#
# \brief The kind of a memory copy, indicating the source and
# destination targets of the copy.
#
# Each kind represents the source and destination targets of a memory
# copy. Targets are host, device, and array.
#
CUpti_ActivityMemcpyKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityMemcpyKind_(IntEnum): 
  #
  # The memory copy kind is not known.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0
  #
  # A host to device memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOD    = 1
  #
  # A device to host memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOH    = 2
  #
  # A host to device array memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOA    = 3
  #
  # A device array to host memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOH    = 4
  #
  # A device array to device array memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOA    = 5
  #
  # A device array to device memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOD    = 6
  #
  # A device to device array memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOA    = 7
  #
  # A device to device memory copy on the same device.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOD    = 8
  #
  # A host to host memory copy.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOH    = 9
  #
  # A peer to peer memory copy across different devices.
  #
  CUPTI_ACTIVITY_MEMCPY_KIND_PTOP    = 10

  CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 0x7fffffff

#
# \brief The kinds of memory accessed by a memory operation/copy.
#
# Each kind represents the type of the memory
# accessed by a memory operation/copy.
#
CUpti_ActivityMemoryKind = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityMemoryKind_(IntEnum): 
  #
  # The memory kind is unknown.
  #
  CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN            = 0
  #
  # The memory is pageable.
  #
  CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE           = 1
  #
  # The memory is pinned.
  #
  CUPTI_ACTIVITY_MEMORY_KIND_PINNED             = 2
  #
  # The memory is on the device.
  #
  CUPTI_ACTIVITY_MEMORY_KIND_DEVICE             = 3
  #
  # The memory is an array.
  #
  CUPTI_ACTIVITY_MEMORY_KIND_ARRAY              = 4
  #
  # The memory is managed
  #
  CUPTI_ACTIVITY_MEMORY_KIND_MANAGED            = 5
  #
  # The memory is device static
  #
  CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC      = 6
  #
  # The memory is managed static
  #
  CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC     = 7
  CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT          = 0x7fffffff

#
#  \brief Partitioned global caching option
#
CUpti_ActivityPartitionedGlobalCacheConfig = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityPartitionedGlobalCacheConfig_(IntEnum): 
  #
  #  Partitioned global cache config unknown.
  #
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN = 0
  #
  #  Partitioned global cache not supported.
  #
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED = 1
  #
  #  Partitioned global cache config off.
  #
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF = 2
  #
  #  Partitioned global cache config on.
  #
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON = 3
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT  = 0x7fffffff


CUpti_ChannelType = ctypes.c_int 
@export(globals()) 
class CUpti_ChannelType_(IntEnum): 
  CUPTI_CHANNEL_TYPE_INVALID = 0
  CUPTI_CHANNEL_TYPE_COMPUTE = 1
  CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY = 2

#
# \brief PC sampling configuration structure
#
# This structure defines the pc sampling configuration.
#
# See function \ref cuptiActivityConfigurePCSampling
#
class CUpti_ActivityPCSamplingConfig(ctypes.Structure): 
    _fields_ = [#
                # Size of configuration structure.
                # CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
                # available in the structure. Used to preserve backward compatibility.
                #
                ("size", ctypes.c_uint32), 
                #
                # There are 5 level provided for sampling period. The level
                # internally maps to a period in terms of cycles. Same level can
                # map to different number of cycles on different gpus. No of
                # cycles will be chosen to minimize information loss. The period
                # chosen will be given by samplingPeriodInCycles in
                # \ref CUpti_ActivityPCSamplingRecordInfo for each kernel instance.
                #
                ("samplingPeriod", CUpti_ActivityPCSamplingPeriod), 
                #
                # This will override the period set by samplingPeriod. Value 0 in samplingPeriod2 will be
                # considered as samplingPeriod2 should not be used and samplingPeriod should be used.
                # Valid values for samplingPeriod2 are between 5 to 31 both inclusive.
                # This will set the sampling period to (2^samplingPeriod2) cycles.
                #
                ("samplingPeriod2", ctypes.c_uint32) 
    ]

#
#  \brief The base activity record.
# 
#  The activity API uses a CUpti_Activity as a generic representation
#  for any activity. The 'kind' field is used to determine the
#  specific activity kind, and from that the CUpti_Activity object can
#  be cast to the specific activity record type appropriate for that kind.
# 
#  Note that all activity record types are padded and aligned to
#  ensure that each member of the record is naturally aligned.
# 
#  \see CUpti_ActivityKind
#
class CUpti_Activity(ctypes.Structure): 
    _fields_ = [#
                # The kind of this activity. 
                #
                ("kind", CUpti_ActivityKind) 
    ]

#
#  \brief The activity record for memory copies. 
# 
#  This activity record represents a memory copy
#  (CUPTI_ACTIVITY_KIND_MEMCPY).
#
class CUpti_ActivityMemcpy5(ctypes.Structure): 
    _fields_ = [#
                #  The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
                #
                ("kind", CUpti_ActivityKind), 

                #
                #  The kind of the memory copy, stored as a byte to reduce record
                #  size. \see CUpti_ActivityMemcpyKind
                #
                ("copyKind", ctypes.c_uint8), 

                #
                #  The source memory kind read by the memory copy, stored as a byte
                #  to reduce record size. \see CUpti_ActivityMemoryKind
                #
                ("srcKind", ctypes.c_uint8), 

                #
                #  The destination memory kind read by the memory copy, stored as a
                #  byte to reduce record size. \see CUpti_ActivityMemoryKind
                #
                ("dstKind", ctypes.c_uint8), 

                #
                #  The flags associated with the memory copy. \see CUpti_ActivityFlag
                #
                ("flags", ctypes.c_uint8), 

                #
                #  The number of bytes transferred by the memory copy.
                #
                ("bytes", ctypes.c_uint64), 

                #
                #  The start timestamp for the memory copy, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the memory copy.
                #
                ("start", ctypes.c_uint64), 

                #
                #  The end timestamp for the memory copy, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the memory copy.
                #
                ("end", ctypes.c_uint64), 

                #
                #  The ID of the device where the memory copy is occurring.
                #
                ("deviceId", ctypes.c_uint32), 

                #
                #  The ID of the context where the memory copy is occurring.
                #
                ("contextId", ctypes.c_uint32), 

                #
                #  The ID of the stream where the memory copy is occurring.
                #
                ("streamId", ctypes.c_uint32), 

                #
                #  The correlation ID of the memory copy. Each memory copy is
                #  assigned a unique correlation ID that is identical to the
                #  correlation ID in the driver API activity record that launched
                #  the memory copy.
                #
                ("correlationId", ctypes.c_uint32), 

                #
                #  The runtime correlation ID of the memory copy. Each memory copy
                #  is assigned a unique runtime correlation ID that is identical to
                #  the correlation ID in the runtime API activity record that
                #  launched the memory copy.
                #
                ("runtimeCorrelationId", ctypes.c_uint32), 

                #ifdef CUPTILP64
                #
                #  Undefined. Reserved for internal use.
                #
                ("pad", ctypes.c_uint32), 
                #endif

                #
                #  Undefined. Reserved for internal use.
                #
                ("reserved0", ctypes.c_void_p), 

                #
                # The unique ID of the graph node that executed this memcpy through graph launch.
                # This field will be 0 if the memcpy is not done through graph launch.
                #
                ("graphNodeId", ctypes.c_uint64), 

                #
                # The unique ID of the graph that executed this memcpy through graph launch.
                # This field will be 0 if the memcpy is not done through graph launch.
                #
                ("graphId", ctypes.c_uint32), 

                #
                # The ID of the HW channel on which the memory copy is occuring.
                #
                ("channelID", ctypes.c_uint32), 

                #
                # The type of the channel
                #
                ("channelType", CUpti_ChannelType), 

                #
                #  Reserved for internal use.
                #
                ("pad2", ctypes.c_uint32) 
    ]

#
#  \brief The activity record for memset. 
# 
#  This activity record represents a memory set operation
#  (CUPTI_ACTIVITY_KIND_MEMSET).
#
class CUpti_ActivityMemset4(ctypes.Structure): 
    _fields_ = [#
                #  The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
                #
                ("kind", CUpti_ActivityKind), 

                #
                #  The value being assigned to memory by the memory set.
                #
                ("value", ctypes.c_uint32), 

                #
                #  The number of bytes being set by the memory set.
                #
                ("bytes", ctypes.c_uint64), 

                #
                #  The start timestamp for the memory set, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the memory set.
                #
                ("start", ctypes.c_uint64), 

                #
                #  The end timestamp for the memory set, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the memory set.
                #
                ("end", ctypes.c_uint64), 

                #
                #  The ID of the device where the memory set is occurring.
                #
                ("deviceId", ctypes.c_uint32), 

                #
                #  The ID of the context where the memory set is occurring.
                #
                ("contextId", ctypes.c_uint32), 

                #
                #  The ID of the stream where the memory set is occurring.
                #
                ("streamId", ctypes.c_uint32), 

                #
                #  The correlation ID of the memory set. Each memory set is assigned
                #  a unique correlation ID that is identical to the correlation ID
                #  in the driver API activity record that launched the memory set.
                #
                ("correlationId", ctypes.c_uint32), 

                #
                #  The flags associated with the memset. \see CUpti_ActivityFlag
                #
                ("flags", ctypes.c_uint16), 

                #
                #  The memory kind of the memory set \see CUpti_ActivityMemoryKind
                #
                ("memoryKind", ctypes.c_uint16), 

                #ifdef CUPTILP64
                #
                #  Undefined. Reserved for internal use.
                #
                ("pad", ctypes.c_uint32), 
                #endif

                #
                #  Undefined. Reserved for internal use.
                #
                ("reserved0", ctypes.c_void_p), 

                #
                # The unique ID of the graph node that executed this memcpy through graph launch.
                # This field will be 0 if the memcpy is not done through graph launch.
                #
                ("graphNodeId", ctypes.c_uint64), 

                #
                # The unique ID of the graph that executed this memcpy through graph launch.
                # This field will be 0 if the memcpy is not done through graph launch.
                #
                ("graphId", ctypes.c_uint32), 

                #
                # The ID of the HW channel on which the memory copy is occuring.
                #
                ("channelID", ctypes.c_uint32), 

                #
                # The type of the channel
                #
                ("channelType", CUpti_ChannelType), 

                #
                #  Reserved for internal use.
                #
                ("pad2", ctypes.c_uint32) 
    ]

#
# \brief The shared memory limit per block config for a kernel
# This should be used to set 'cudaOccFuncShmemConfig' field in occupancy calculator API
#
CUpti_FuncShmemLimitConfig = ctypes.c_int 
@export(globals()) 
class CUpti_FuncShmemLimitConfig_(IntEnum): 
  ''' The shared memory limit config is default '''
  CUPTI_FUNC_SHMEM_LIMIT_DEFAULT              = 0x00
  ''' User has opted for a higher dynamic shared memory limit using function attribute
      'cudaFuncAttributeMaxDynamicSharedMemorySize' for runtime API or
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for driver API '''
  CUPTI_FUNC_SHMEM_LIMIT_OPTIN                = 0x01
  CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT            = 0x7fffffff

#
#  \brief The activity record for a kernel 
# 
#  This activity record represents a kernel execution
#  (CUPTI_ACTIVITY_KIND_KERNEL and
#  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) 
#
class _config(ctypes.Structure): 
    _fields_ = [#
                #  The cache configuration requested by the kernel. The value is one
                #  of the CUfunc_cache enumeration values from cuda.h.
                #
                ("requested", ctypes.c_uint8, 4), 
                #
                #  The cache configuration used for the kernel. The value is one of
                #  the CUfunc_cache enumeration values from cuda.h.
                #
               ("executed", ctypes.c_uint8, 4) 
    ]
class _cacheConfig(ctypes.Union): 
   _fields_ = [("both", ctypes.c_uint8), 
               ("config", _config), 
      
   ]

class CUpti_ActivityKernel7(ctypes.Structure): 
    _fields_ = [#
                #  The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
                #  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
                #
                ("kind", CUpti_ActivityKind), 

                #
                #  For devices with compute capability 7.0+ cacheConfig values are not updated
                #  in case field isSharedMemoryCarveoutRequested is set
                #
                ("cacheConfig", _cacheConfig), 

                #
                #  The shared memory configuration used for the kernel. The value is one of
                #  the CUsharedconfig enumeration values from cuda.h.
                #
                ("sharedMemoryConfig", ctypes.c_uint8), 

                #
                #  The number of registers required for each thread executing the
                #  kernel.
                #
                ("registersPerThread", ctypes.c_uint16), 

                #
                #  The partitioned global caching requested for the kernel. Partitioned
                #  global caching is required to enable caching on certain chips, such as
                #  devices with compute capability 5.2.
                #
                ("partitionedGlobalCacheRequested", CUpti_ActivityPartitionedGlobalCacheConfig), 

                #
                #  The partitioned global caching executed for the kernel. Partitioned
                #  global caching is required to enable caching on certain chips, such as
                #  devices with compute capability 5.2. Partitioned global caching can be
                #  automatically disabled if the occupancy requirement of the launch cannot
                #  support caching.
                #
                ("partitionedGlobalCacheExecuted", CUpti_ActivityPartitionedGlobalCacheConfig), 

                #
                #  The start timestamp for the kernel execution, in ns. A value of 0
                #  for both the start and end timestamps indicates that timestamp
                #  information could not be collected for the kernel.
                #
                ("start", ctypes.c_uint64), 

                #
                #  The end timestamp for the kernel execution, in ns. A value of 0
                #  for both the start and end timestamps indicates that timestamp
                #  information could not be collected for the kernel.
                #
                ("end", ctypes.c_uint64), 

                #
                #  The completed timestamp for the kernel execution, in ns.  It
                #  represents the completion of all it's child kernels and the
                #  kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
                #  the completion time is unknown.
                #
                ("completed", ctypes.c_uint64), 

                #
                #  The ID of the device where the kernel is executing.
                #
                ("deviceId", ctypes.c_uint32), 

                #
                #  The ID of the context where the kernel is executing.
                #
                ("contextId", ctypes.c_uint32), 

                #
                #  The ID of the stream where the kernel is executing.
                #
                ("streamId", ctypes.c_uint32), 

                #
                #  The X-dimension grid size for the kernel.
                #
                ("gridX", ctypes.c_int32), 

                #
                #  The Y-dimension grid size for the kernel.
                #
                ("gridY", ctypes.c_int32), 

                #
                #  The Z-dimension grid size for the kernel.
                #
                ("gridZ", ctypes.c_int32), 

                #
                #  The X-dimension block size for the kernel.
                #
                ("blockX", ctypes.c_int32), 

                #
                #  The Y-dimension block size for the kernel.
                #
                ("blockY", ctypes.c_int32), 

                #
                #  The Z-dimension grid size for the kernel.
                #
                ("blockZ", ctypes.c_int32), 

                #
                #  The static shared memory allocated for the kernel, in bytes.
                #
                ("staticSharedMemory", ctypes.c_int32), 

                #
                #  The dynamic shared memory reserved for the kernel, in bytes.
                #
                ("dynamicSharedMemory", ctypes.c_int32), 

                #
                #  The amount of local memory reserved for each thread, in bytes.
                #
                ("localMemoryPerThread", ctypes.c_uint32), 

                #
                #  The total amount of local memory reserved for the kernel, in
                #  bytes.
                #
                ("localMemoryTotal", ctypes.c_uint32), 

                #
                #  The correlation ID of the kernel. Each kernel execution is
                #  assigned a unique correlation ID that is identical to the
                #  correlation ID in the driver or runtime API activity record that
                #  launched the kernel.
                #
                ("correlationId", ctypes.c_uint32), 

                #
                #  The grid ID of the kernel. Each kernel is assigned a unique
                #  grid ID at runtime.
                #
                ("gridId", ctypes.c_int64), 

                #
                #  The name of the kernel. This name is shared across all activity
                #  records representing the same kernel, and so should not be
                #  modified.
                #
                ("name", ctypes.c_char_p), 

                #
                #  Undefined. Reserved for internal use.
                #
                ("reserved0", ctypes.c_void_p), 

                #
                #  The timestamp when the kernel is queued up in the command buffer, in ns.
                #  A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
                #  could not be collected for the kernel. This timestamp is not collected
                #  by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
                #  enable collection.
                # 
                #  Command buffer is a buffer written by CUDA driver to send commands
                #  like kernel launch, memory copy etc to the GPU. All launches of CUDA
                #  kernels are asynchrnous with respect to the host, the host requests
                #  the launch by writing commands into the command buffer, then returns
                #  without checking the GPU's progress.
                #
                ("queued", ctypes.c_uint64), 

                #
                #  The timestamp when the command buffer containing the kernel launch
                #  is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
                #  indicates that the submitted time could not be collected for the kernel.
                #  This timestamp is not collected by default. Use API \ref
                #  cuptiActivityEnableLatencyTimestamps() to enable collection.
                #
                ("submitted", ctypes.c_uint64), 

                #
                #  The indicates if the kernel was executed via a regular launch or via a
                #  single/multi device cooperative launch. \see CUpti_ActivityLaunchType
                #
                ("launchType", ctypes.c_uint8), 

                #
                #  This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
                #  updated for the kernel launch
                #
                ("isSharedMemoryCarveoutRequested", ctypes.c_uint8), 

                #
                #  Shared memory carveout value requested for the function in percentage of
                #  the total resource. The value will be updated only if field
                #  isSharedMemoryCarveoutRequested is set.
                #
                ("sharedMemoryCarveoutRequested", ctypes.c_uint8), 

                #
                #  Undefined. Reserved for internal use.
                #
                ("padding", ctypes.c_uint8), 

                #
                #  Shared memory size set by the driver.
                #
                ("sharedMemoryExecuted", ctypes.c_uint32), 

                #
                # The unique ID of the graph node that launched this kernel through graph launch APIs.
                # This field will be 0 if the kernel is not launched through graph launch APIs.
                #
                ("graphNodeId", ctypes.c_uint64), 

                #
                # The shared memory limit config for the kernel. This field shows whether user has opted for a
                # higher per block limit of dynamic shared memory.
                #
                ("shmemLimitConfig", CUpti_FuncShmemLimitConfig), 

                #
                # The unique ID of the graph that launched this kernel through graph launch APIs.
                # This field will be 0 if the kernel is not launched through graph launch APIs.
                #
                ("graphId", ctypes.c_uint32), 

                #
                # The pointer to the access policy window. The structure CUaccessPolicyWindow is
                # defined in cuda.h.
                #
                ("pAccessPolicyWindow", ctypes.POINTER(CUaccessPolicyWindow)), 

                #
                # The ID of the HW channel on which the kernel is launched.
                #
                ("channelID", ctypes.c_uint32), 

                #
                # The type of the channel
                #
                ("channelType", CUpti_ChannelType) 
    ]

#
#  \brief The activity record for CUPTI and driver overheads.
# 
#  This activity record provides CUPTI and driver overhead information
#  (CUPTI_ACTIVITY_OVERHEAD).
#
class CUpti_ActivityOverhead(ctypes.Structure): 
    _fields_ = [#
                #  The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
                #
                ("kind", CUpti_ActivityKind), 

                #
                #  The kind of overhead, CUPTI, DRIVER, COMPILER etc.
                #
                ("overheadKind", CUpti_ActivityOverheadKind), 

                #
                #  The kind of activity object that the overhead is associated with.
                #
                ("objectKind", CUpti_ActivityObjectKind), 

                #
                #  The identifier for the activity object. 'objectKind' indicates
                #  which ID is valid for this record.
                #
                ("objectId", CUpti_ActivityObjectKindId), 

                #
                #  The start timestamp for the overhead, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the overhead.
                #
                ("start", ctypes.c_uint64), 

                #
                #  The end timestamp for the overhead, in ns. A value of 0 for both
                #  the start and end timestamps indicates that timestamp information
                #  could not be collected for the overhead.
                #
                ("end", ctypes.c_uint64) 
    ]

#
#  \brief The activity record for a driver or runtime API invocation.
# 
#  This activity record represents an invocation of a driver or
#  runtime API (CUPTI_ACTIVITY_KIND_DRIVER and
#  CUPTI_ACTIVITY_KIND_RUNTIME).
#
class CUpti_ActivityAPI(ctypes.Structure): 
    _fields_ = [#
                #  The activity record kind, must be CUPTI_ACTIVITY_KIND_DRIVER,
                #  CUPTI_ACTIVITY_KIND_RUNTIME, or CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API.
                #
                ("kind", CUpti_ActivityKind), 

                #
                #  The ID of the driver or runtime function.
                #
                ("cbid", CUpti_CallbackId), 

                #
                #  The start timestamp for the function, in ns. A value of 0 for
                #  both the start and end timestamps indicates that timestamp
                #  information could not be collected for the function.
                #
                ("start", ctypes.c_uint64), 

                #
                #  The end timestamp for the function, in ns. A value of 0 for both
                #  the start and end timestamps indicates that timestamp information
                #  could not be collected for the function.
                #
                ("end", ctypes.c_uint64), 

                #
                #  The ID of the process where the driver or runtime CUDA function
                #  is executing.
                #
                ("processId", ctypes.c_uint32), 

                #
                #  The ID of the thread where the driver or runtime CUDA function is
                #  executing.
                #
                ("threadId", ctypes.c_uint32), 

                #
                #  The correlation ID of the driver or runtime CUDA function. Each
                #  function invocation is assigned a unique correlation ID that is
                #  identical to the correlation ID in the memcpy, memset, or kernel
                #  activity record that is associated with this function.
                #
                ("correlationId", ctypes.c_uint32), 

                #
                #  The return value for the function. For a CUDA driver function
                #  with will be a CUresult value, and for a CUDA runtime function
                #  this will be a cudaError_t value.
                #
                ("returnValue", ctypes.c_uint32) 
    ]

#
#  \brief Activity attributes.
# 
#  These attributes are used to control the behavior of the activity
#  API.
#
CUpti_ActivityAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_ActivityAttribute_(IntEnum): 
  #
  #  The device memory size (in bytes) reserved for storing profiling data for concurrent
  #  kernels (activity kind \ref CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), memcopies and memsets
  #  for each buffer on a context. The value is a size_t.
  # 
  #  There is a limit on how many device buffers can be allocated per context. User
  #  can query and set this limit using the attribute
  #  \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
  #  CUPTI doesn't pre-allocate all the buffers, it pre-allocates only those many
  #  buffers as set by the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
  #  When all of the data in a buffer is consumed, it is added in the reuse pool, and
  #  CUPTI picks a buffer from this pool when a new buffer is needed. Thus memory
  #  footprint does not scale with the kernel count. Applications with the high density
  #  of kernels, memcopies and memsets might result in having CUPTI to allocate more device buffers.
  #  CUPTI allocates another buffer only when it runs out of the buffers in the
  #  reuse pool.
  # 
  #  Since buffer allocation happens in the main application thread, this might result
  #  in stalls in the critical path. CUPTI pre-allocates 3 buffers of the same size to
  #  mitigate this issue. User can query and set the pre-allocation limit using the
  #  attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
  # 
  #  Having larger buffer size leaves less device memory for the application.
  #  Having smaller buffer size increases the risk of dropping timestamps for
  #  records if too many kernels or memcopies or memsets are launched at one time.
  # 
  #  This value only applies to new buffer allocations. Set this value before initializing
  #  CUDA or before creating a context to ensure it is considered for the following allocations.
  # 
  #  The default value is 3200000 (~3MB) which can accommodate profiling data
  #  up to 100,000 kernels, memcopies and memsets combined.
  # 
  #  Note: Starting with the CUDA 11.2 release, CUPTI allocates profiling buffer in the
  #  pinned host memory by default as this might help in improving the performance of the
  #  tracing run. Refer to the description of the attribute
  #  \ref CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED for more details.
  #  Size of the memory and maximum number of pools are still controlled by the attributes
  #  \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE and \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
  # 
  #  Note: The actual amount of device memory per buffer reserved by CUPTI might be larger.
  #
  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE                      = 0
  #
  #  The device memory size (in bytes) reserved for storing profiling
  #  data for CDP operations for each buffer on a context. The
  #  value is a size_t.
  # 
  #  Having larger buffer size means less flush operations but
  #  consumes more device memory. This value only applies to new
  #  allocations.
  # 
  #  Set this value before initializing CUDA or before creating a
  #  context to ensure it is considered for the following allocations.
  # 
  #  The default value is 8388608 (8MB).
  # 
  #  Note: The actual amount of device memory per context reserved by
  #  CUPTI might be larger.
  #
  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP          = 1
  #
  #  The maximum number of device memory buffers per context. The value is a size_t.
  # 
  #  For an application with high rate of kernel launches, memcopies and memsets having a bigger pool
  #  limit helps in timestamp collection for all these activties at the expense of a larger memory footprint.
  #  Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
  #  for more details.
  # 
  #  Setting this value will not modify the number of memory buffers
  #  currently stored.
  # 
  #  Set this value before initializing CUDA to ensure the limit is
  #  not exceeded.
  # 
  #  The default value is 250.
  #
  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT                = 2

  #
  #  The profiling semaphore pool size reserved for storing profiling data for
  #  serialized kernels tracing (activity kind \ref CUPTI_ACTIVITY_KIND_KERNEL)
  #  for each context. The value is a size_t.
  # 
  #  There is a limit on how many semaphore pools can be allocated per context. User
  #  can query and set this limit using the attribute
  #  \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT.
  #  CUPTI doesn't pre-allocate all the semaphore pools, it pre-allocates only those many
  #  semaphore pools as set by the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
  #  When all of the data in a semaphore pool is consumed, it is added in the reuse pool, and
  #  CUPTI picks a semaphore pool from the reuse pool when a new semaphore pool is needed. Thus memory
  #  footprint does not scale with the kernel count. Applications with the high density
  #  of kernels might result in having CUPTI to allocate more semaphore pools.
  #  CUPTI allocates another semaphore pool only when it runs out of the semaphore pools in the
  #  reuse pool.
  # 
  #  Since semaphore pool allocation happens in the main application thread, this might result
  #  in stalls in the critical path. CUPTI pre-allocates 3 semaphore pools of the same size to
  #  mitigate this issue. User can query and set the pre-allocation limit using the
  #  attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
  # 
  #  Having larger semaphore pool size leaves less device memory for the application.
  #  Having smaller semaphore pool size increases the risk of dropping timestamps for
  #  kernel records if too many kernels are issued/launched at one time.
  # 
  #  This value only applies to new semaphore pool allocations. Set this value before initializing
  #  CUDA or before creating a context to ensure it is considered for the following allocations.
  # 
  #  The default value is 25000 which can accommodate profiling data for upto 25,000 kernels.
  # 
  #
  CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE           = 3
  #
  #  The maximum number of profiling semaphore pools per context. The value is a size_t.
  # 
  #  For an application with high rate of kernel launches, having a bigger
  #  pool limit helps in timestamp collection for all the kernels, at the
  #  expense of a larger device memory footprint.
  #  Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
  #  for more details.
  # 
  #  Set this value before initializing CUDA to ensure the limit is not exceeded.
  # 
  #  The default value is 250.
  #
  CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT          = 4

  #
  #  The flag to indicate whether user should provide activity buffer of zero value.
  #  The value is a uint8_t.
  # 
  #  If the value of this attribute is non-zero, user should provide
  #  a zero value buffer in the \ref CUpti_BuffersCallbackRequestFunc.
  #  If the user does not provide a zero value buffer after setting this to non-zero,
  #  the activity buffer may contain some uninitialized values when CUPTI returns it in
  #  \ref CUpti_BuffersCallbackCompleteFunc
  # 
  #  If the value of this attribute is zero, CUPTI will initialize the user buffer
  #  received in the \ref CUpti_BuffersCallbackRequestFunc to zero before filling it.
  #  If the user sets this to zero, a few stalls may appear in critical path because CUPTI
  #  will zero out the buffer in the main thread.
  #  Set this value before returning from \ref CUpti_BuffersCallbackRequestFunc to
  #  ensure it is considered for all the subsequent user buffers.
  # 
  #  The default value is 0.
  #
  CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER              = 5

  #
  #  Number of device buffers to pre-allocate for a context during the initialization phase.
  #  The value is a size_t.
  # 
  #  Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
  #  for details.
  # 
  #  This value must be less than the maximum number of device buffers set using
  #  the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT
  # 
  #  Set this value before initializing CUDA or before creating a context to ensure it
  #  is considered by the CUPTI.
  # 
  #  The default value is set to 3 to ping pong between these buffers (if possible).
  #
  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE        = 6

  #
  #  Number of profiling semaphore pools to pre-allocate for a context during the
  #  initialization phase. The value is a size_t.
  # 
  #  Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
  #  for details.
  # 
  #  This value must be less than the maximum number of profiling semaphore pools set
  #  using the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT
  # 
  #  Set this value before initializing CUDA or before creating a context to ensure it
  #  is considered by the CUPTI.
  # 
  #  The default value is set to 3 to ping pong between these pools (if possible).
  #
  CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE  = 7

  #
  #  Allocate page-locked (pinned) host memory for storing profiling data for concurrent
  #  kernels, memcopies and memsets for each buffer on a context. The value is a uint8_t.
  # 
  #  Starting with the CUDA 11.2 release, CUPTI allocates profiling buffer in the pinned host
  #  memory by default as this might help in improving the performance of the tracing run.
  #  Allocating excessive amounts of pinned memory may degrade system performance, since it
  #  reduces the amount of memory available to the system for paging. For this reason user
  #  might want to change the location from pinned host memory to device memory by setting
  #  value of this attribute to 0.
  # 
  #  The default value is 1.
  #
  CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED         = 8


  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT                 = 0x7fffffff

#
# \brief Function type for callback used by CUPTI to request an empty
# buffer for storing activity records.
#
# This callback function signals the CUPTI client that an activity
# buffer is needed by CUPTI. The activity buffer is used by CUPTI to
# store activity records. The callback function can decline the
# request by setting \p *buffer to NULL. In this case CUPTI may drop
# activity records.
#
# \param buffer Returns the new buffer. If set to NULL then no buffer
# is returned.
# \param size Returns the size of the returned buffer.
# \param maxNumRecords Returns the maximum number of records that
# should be placed in the buffer. If 0 then the buffer is filled with
# as many records as possible. If > 0 the buffer is filled with at
# most that many records before it is returned.
#
CUpti_BuffersCallbackRequestFunc = ctypes.c_void_p 

#
#  \brief Function type for callback used by CUPTI to return a buffer
#  of activity records.
# 
#  This callback function returns to the CUPTI client a buffer
#  containing activity records.  The buffer contains \p validSize
#  bytes of activity records which should be read using
#  cuptiActivityGetNextRecord. The number of dropped records can be
#  read using cuptiActivityGetNumDroppedRecords. After this call CUPTI
#  relinquished ownership of the buffer and will not use it
#  anymore. The client may return the buffer to CUPTI using the
#  CUpti_BuffersCallbackRequestFunc callback.
#  Note: CUDA 6.0 onwards, all buffers returned by this callback are
#  global buffers i.e. there is no context/stream specific buffer.
#  User needs to parse the global buffer to extract the context/stream
#  specific activity records.
# 
#  \param context The context this buffer is associated with. If NULL, the
#  buffer is associated with the global activities. This field is deprecated
#  as of CUDA 6.0 and will always be NULL.
#  \param streamId The stream id this buffer is associated with.
#  This field is deprecated as of CUDA 6.0 and will always be NULL.
#  \param buffer The activity record buffer.
#  \param size The total size of the buffer in bytes as set in
#  CUpti_BuffersCallbackRequestFunc.
#  \param validSize The number of valid bytes in the buffer.
#
CUpti_BuffersCallbackCompleteFunc = ctypes.c_void_p 