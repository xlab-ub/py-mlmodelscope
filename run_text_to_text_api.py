import argparse

from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import uvicorn 

from mlmodelscope import MLModelScope

TRACE_LEVEL = ( "NO_TRACE",
                "APPLICATION_TRACE",
                "MODEL_TRACE",          # pipelines within model
                "FRAMEWORK_TRACE",      # layers within framework
                "ML_LIBRARY_TRACE",     # cudnn, ...
                "SYSTEM_LIBRARY_TRACE", # cupti
                "HARDWARE_TRACE",       # perf, papi, ...
                "FULL_TRACE")           # includes all of the above)

app = FastAPI()

class QueryModelRequest(BaseModel):
    model: str
    messages: List[dict]  # Expected format: [{"content": "message content"}]

class ModelServer:
    def __init__(self):
        self.mlms = None
        self.agent = None
    
    def load_mlmodelscope(self, architecture, trace_level, gpu_trace):
        """Loads the model and initializes MLModelScope."""
        self.mlms = MLModelScope(architecture, trace_level, gpu_trace)
    
    async def query_model(self, data: QueryModelRequest):
        """Endpoint to handle queries to the model."""
        if not data.model or not data.messages:
            error_message = "No model provided!" if not data.model else "No messages provided!"
            raise HTTPException(status_code=400, detail={"error": error_message})
        
        # Load the model (agent)
        self.mlms.load_agent("text_to_text", self.agent, data.model, False, data.model_dump(), "default")
        
        # Create input data for the model
        input_data = [data.messages[-1].get('content', '')]
        self.mlms.load_dataset_api(input_data, batch_size=1, task='text_to_text', security_check=False)

        # Perform prediction
        outputs = self.mlms.predict(0, False)
        
        # Build response
        response = {"choices": [{"index": 0, "message": {"role": "assistant", "content": outputs[0]}}]}
        return response

model_server = ModelServer()

@app.post('/api/chat')
async def handle_chat(data: QueryModelRequest):
    return await model_server.query_model(data)

def main():
    parser = argparse.ArgumentParser(description="mlmodelscope")
    parser.add_argument("--agent", type=str, nargs='?', default="pytorch", choices=["pytorch", "tensorflow", "onnxruntime", "mxnet", "jax"], help="Which framework to use") 
    parser.add_argument('--port', type=int, default=15555, help="Port to run the server on") 
    parser.add_argument("--architecture", type=str, nargs='?', default="gpu", choices=["cpu", "gpu"], help="Which Processing Unit to use") 
    parser.add_argument("--trace_level", type=str, nargs='?', default="NO_TRACE", choices=TRACE_LEVEL, help="MLModelScope Trace Level") 
    parser.add_argument("--gpu_trace", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to trace GPU activities") 
        
    args = parser.parse_args()

    trace_level = args.trace_level
    gpu_trace = True if (trace_level != "NO_TRACE") and (args.gpu_trace == "true") else False

    # Load the model before starting the server
    model_server.load_mlmodelscope(args.architecture, trace_level, gpu_trace)
    model_server.agent = args.agent

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()