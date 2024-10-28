from .dataset_abc import DatasetAbstractClass 
import inspect 
import ast 
import os 
import pathlib 

def check_for_unsafe_modules(module_name):
  unsafe_modules = ['os', 'sys', 'subprocess']
  return module_name in unsafe_modules

def check_for_unsafe_functions(function_name):
  unsafe_functions = ['eval', 'pickle', 'exec']
  return function_name in unsafe_functions

def perform_syntax_and_security_check(file_name):
  '''
  Perform security check on the dataset file.

  '''
  with open(file_name, 'r') as file:
    try:
      # Parse the file into an AST (Abstract Syntax Tree)
      tree = ast.parse(file.read())

      # Check if the AST contains any dangerous nodes
      unsafe_nodes = []
      for node in ast.walk(tree):
        if isinstance(node, ast.Import):
          for name in node.names:
            if check_for_unsafe_modules(name.name):
              unsafe_nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
          if node.module:
            if check_for_unsafe_modules(node.module):
              unsafe_nodes.append(node)
        elif isinstance(node, ast.Call):
          if hasattr(node.func, 'id'):
            if check_for_unsafe_functions(node.func.id):
              unsafe_nodes.append(node)

      if len(unsafe_nodes) > 0:
        print("Unsafe code detected:")
        for node in unsafe_nodes:
          print(f"Found unsafe call: {ast.dump(node)}")
        return False
      else:
        return True

    except SyntaxError as e:
      raise Exception(f"Syntax Error in the file: {e}")

def create_instance_from_dataset_manifest_file(dataset_name, security_check=True):
  '''
  Create an instance of a class from a file.
  '''
  file_name = os.path.join(pathlib.Path(__file__).resolve().parent, f'datasets/{dataset_name}.py') # Get the path of the dataset file
  if security_check and (not perform_syntax_and_security_check(file_name)):
    raise Exception("Security issue detected. Aborting.")

  with open(file_name, 'r') as file:
    code = compile(file.read(), file_name, 'exec')
    
    globals_dict = {
      '__name__': '.'.join(__name__.split('.')[:-1]) + '.datasets' + '.', 
      '__file__': file_name, 
    }

    exec(code, globals_dict)  # Execute the code in the current scope

  # Get all classes in the current scope
  classes = [cls for cls in globals_dict.values() if inspect.isclass(cls)]

  # Find the class that inherits from DatasetAbstractClass
  abstractClass = DatasetAbstractClass 
  target_class = None

  for cls in classes:
    if issubclass(cls, abstractClass) and cls != abstractClass:
      target_class = cls
      break

  if target_class:
    instance = target_class()  # Create an instance of the found class
    return instance
  else:
    raise ModuleNotFoundError("No subclass of DatasetAbstractClass was found in the module.") 


def load(dataset_name, url=False, count=None, task=None, security_check=True):
  try:
    if url: 
      if isinstance(dataset_name[0], dict):
        dataset_name = [url['src'] for url in dataset_name]
      exec('from .datasets.url_data' + ' import Url_Data', globals())
      return Url_Data(dataset_name) 
    else: 
      if task is None:
        exec('from .datasets.' + dataset_name + ' import init', globals())
        if (count is not None) and (count > 0):
          return init(count)
        else: 
          return init() 
      elif task[:4] == "text": 
        exec('from .datasets.text_data import Text_Data', globals())
        return Text_Data(dataset_name) 
      else:
        raise NotImplementedError(f"{task} dataset is not supported")
    print(f"{dataset_name} dataset exists")
    print(url)
  except ImportError as e:
    if e.msg.split()[3] == "'init'": 
      return create_instance_from_dataset_manifest_file(dataset_name, security_check) 