def load(dataset_name, url=False, count=None):
  if url: 
    exec('from .datasets.url_data' + ' import init', globals())
    return init(dataset_name) 
  else: 
    exec('from .datasets.' + dataset_name + ' import init', globals())
    if (count is not None) and (count > 0):
      return init(count)
    else: 
      return init() 
