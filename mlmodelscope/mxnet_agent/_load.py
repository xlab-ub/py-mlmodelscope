def _load(task, model_name, architecture):
  exec(f'from .models.{task}.' + model_name + ' import init', globals())
  return init(architecture) 
