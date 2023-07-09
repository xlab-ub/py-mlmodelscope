def _load(task, model_name, providers):
  exec(f'from .models.{task}.' + model_name + ' import init', globals())
  return init(providers) 
