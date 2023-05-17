def load(dataset_name):
  exec('from .datasets.' + dataset_name + ' import init', globals())
  return init() 
