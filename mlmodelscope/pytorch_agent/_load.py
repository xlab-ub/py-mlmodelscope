def _load(model_name):
  exec('from .' + model_name + ' import init', globals())
  return init()
