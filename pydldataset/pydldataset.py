def load(dataset_name, url=False):
  if url: 
    exec('from .datasets.url_data' + ' import init', globals())
    return init(dataset_name) 
  else: 
    exec('from .datasets.' + dataset_name + ' import init', globals())
    return init() 
