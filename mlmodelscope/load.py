def load(model_name='alexnet', backend='pytorch'): 
    if backend == 'pytorch': 
        from .pytorch_agent import _load 
        return _load(model_name) 
    elif backend == 'tensorflow': 
        raise NotImplementedError(f'{backend} models are not supported yet!') 
    elif backend == 'onnxruntime': 
        raise NotImplementedError(f'{backend} models are not supported yet!') 
    else: 
        raise NotImplementedError(f'{backend} models are not supported!') 