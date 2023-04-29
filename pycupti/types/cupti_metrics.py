import ctypes 
from aenum import IntEnum, export 

#
# \brief ID for a metric.
#
# A metric provides a measure of some aspect of the device.
#
CUpti_MetricID = ctypes.c_int 

#
# \brief Metric attributes.
#
# Metric attributes describe properties of a metric. These attributes
# can be read using \ref cuptiMetricGetAttribute.
#
CUpti_MetricAttribute = ctypes.c_int 
@export(globals()) 
class CUpti_MetricAttribute_(IntEnum): 
  #
  # Metric name. Value is a null terminated const c-string.
  #
  CUPTI_METRIC_ATTR_NAME              = 0
  #
  # Short description of metric. Value is a null terminated const c-string.
  #
  CUPTI_METRIC_ATTR_SHORT_DESCRIPTION = 1
  #
  # Long description of metric. Value is a null terminated const c-string.
  #
  CUPTI_METRIC_ATTR_LONG_DESCRIPTION  = 2
  #
  # Category of the metric. Value is of type CUpti_MetricCategory.
  #
  CUPTI_METRIC_ATTR_CATEGORY          = 3
  #
  # Value type of the metric. Value is of type CUpti_MetricValueKind.
  #
  CUPTI_METRIC_ATTR_VALUE_KIND          = 4
  #
  # Metric evaluation mode. Value is of type CUpti_MetricEvaluationMode.
  #
  CUPTI_METRIC_ATTR_EVALUATION_MODE     = 5
  CUPTI_METRIC_ATTR_FORCE_INT         = 0x7fffffff
