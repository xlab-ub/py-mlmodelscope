def _load(task, model_name):
  exec(f'from .models.{task}.' + model_name + ' import init', globals())
  return init()
