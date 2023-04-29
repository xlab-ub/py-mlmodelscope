def load(dataset_name, root):
  exec('from .' + dataset_name + ' import init', globals())
  return init(root)
