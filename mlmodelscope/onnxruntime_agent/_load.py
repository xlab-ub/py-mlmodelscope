from .models.onnxruntime_abc import ONNXRuntimeAbstractClass
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
  Perform security check on the model file.

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

def create_instance_from_model_manifest_file(task, model_name, providers):
  '''
  Create an instance of a class from a file.
  '''
  # parent_dir = pathlib.Path(__file__).resolve().parent.__str__()
  file_name = os.path.join(pathlib.Path(__file__).resolve().parent, f'models/{task}/{model_name}.py') # Get the path of the model file
  
  if not perform_syntax_and_security_check(file_name):
    raise Exception("Security issue detected. Aborting.")

  with open(file_name, 'r') as file:
    code = compile(file.read(), file_name, 'exec')
    
    globals_dict = {
      '__name__': '.'.join(__name__.split('.')[:-1]) + '.models' + '.' + task + '.', 
    }

    exec(code, globals_dict)  # Execute the code in the current scope

  # Get all classes in the current scope
  classes = [cls for cls in globals_dict.values() if inspect.isclass(cls)]

  # Find the class that inherits from ONNXRuntimeAbstractClass 
  abstractClass = ONNXRuntimeAbstractClass  
  target_class = None

  for cls in classes:
    if issubclass(cls, abstractClass) and cls != abstractClass:
      target_class = cls
      break

  if target_class:
    instance = target_class(providers)  # Create an instance of the found class
    return instance
  else:
    raise ModuleNotFoundError("No subclass of ONNXRuntimeAbstractClass was found in the module.") 

def _load(task, model_name, providers):
  try: 
    exec(f'from .models.{task}.' + model_name + ' import init', globals())
    return init(providers)
  except ImportError as e:
    if e.msg.split()[3] == "'init'": 
      return create_instance_from_model_manifest_file(task, model_name, providers) # Create an instance of the model class 
