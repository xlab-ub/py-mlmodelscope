from .models.pytorch_abc import PyTorchAbstractClass
import inspect
import ast
import pathlib
import sys

# Use sets for quicker lookups
UNSAFE_MODULES = {'os', 'sys', 'subprocess'}
UNSAFE_FUNCTIONS = {'eval', 'pickle', 'exec'}

def check_for_unsafe_modules(module_name):
    return module_name in UNSAFE_MODULES

def check_for_unsafe_functions(function_name):
    return function_name in UNSAFE_FUNCTIONS

def perform_syntax_and_security_check(file_name):
    '''
    Perform security check on the model file.
    '''
    try:
        with open(file_name, 'r') as file:
            # Parse the file into an AST (Abstract Syntax Tree)
            tree = ast.parse(file.read())

        # Check if the AST contains any dangerous nodes
        unsafe_nodes = [
            node for node in ast.walk(tree)
            if (isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)) and 
               any(check_for_unsafe_modules(name.name if isinstance(node, ast.Import) else node.module) for name in node.names)
            or (isinstance(node, ast.Call) and hasattr(node.func, 'id') and check_for_unsafe_functions(node.func.id))
        ]

        if unsafe_nodes:
            print("Unsafe code detected:")
            for node in unsafe_nodes:
                print(f"Found unsafe call: {ast.dump(node)}")
            return False
        return True

    except SyntaxError as e:
        raise Exception(f"Syntax Error in the file: {e}")

def create_instance_from_model_manifest_file(task, model_name, security_check=True, config=None, user='default', device='cpu', multi_gpu=False):
    '''
    Create an instance of a class from a file.
    '''
    file_path = pathlib.Path(__file__).resolve().parent / f'models/{user}/{task}/{model_name}/model.py' 
    
    if security_check and not perform_syntax_and_security_check(file_path):
        raise Exception("Security issue detected. Aborting.")

    with open(file_path, 'r') as file:
        code = compile(file.read(), file_path, 'exec')

        globals_dict = {
            '__name__': '.'.join(__name__.split('.')[:-1]) + f'.models.{user}.{task}.{model_name}.',
            '__file__': str(file_path),
        }
        exec(code, globals_dict)  # Execute the code in the current scope

    # Get all classes in the current scope
    classes = [cls for cls in globals_dict.values() if inspect.isclass(cls)]

    # Find the class that inherits from PyTorchAbstractClass
    abstract_class = PyTorchAbstractClass
    target_class = next((cls for cls in classes if issubclass(cls, abstract_class) and cls != abstract_class), None)

    if target_class:
        instance_config = config if config else {}
        instance_config['_device'] = device
        instance_config['_multi_gpu'] = multi_gpu
        
        # Check if __init__ accepts config parameter
        init_signature = inspect.signature(target_class.__init__)
        init_params = list(init_signature.parameters.keys())
        
        # Create instance based on signature (excluding 'self')
        if len(init_params) > 1 or 'config' in init_params:
            # New style: accepts config
            instance = target_class(instance_config)
        else:
            # Old style: no config parameter
            # Manually set _device and _multi_gpu BEFORE instantiation by monkey-patching
            # This ensures they're available when __init__ runs
            original_init = target_class.__init__
            def patched_init(self):
                self._device = device
                self._multi_gpu = multi_gpu
                self._is_dispatched = False
                self._disable_accelerate = True  # Flag to disable Accelerate for old models
                self.config = instance_config
                original_init(self)
            target_class.__init__ = patched_init
            instance = target_class()
            # Restore original __init__
            target_class.__init__ = original_init
        
        return instance
    else:
        raise ModuleNotFoundError("No subclass of PyTorchAbstractClass was found in the module.")

def _load(task, model_name, security_check=True, config=None, user='default', device='cpu', multi_gpu=False):
    model_file_dir = pathlib.Path(__file__).resolve().parent / f'models/{user}/{task}/{model_name}'
    sys.path.append(str(model_file_dir))
    pytorch_abstract_class_instance = create_instance_from_model_manifest_file(task, model_name, security_check, config, user, device, multi_gpu) 
    sys.path.remove(str(model_file_dir))
    return pytorch_abstract_class_instance
