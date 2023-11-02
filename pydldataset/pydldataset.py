def load(dataset_name, url=False, count=None, task=None):
  if url: 
    exec('from .datasets.url_data' + ' import init', globals())
    return init(dataset_name) 
  else: 
    if task is None:
      exec('from .datasets.' + dataset_name + ' import init', globals())
      if (count is not None) and (count > 0):
        return init(count)
      else: 
        return init() 
    elif task[:4] == "text": 
      exec('from .datasets.text_data import init', globals())
      return init(dataset_name) 
    else:
      raise NotImplementedError(f"{task} dataset is not supported")