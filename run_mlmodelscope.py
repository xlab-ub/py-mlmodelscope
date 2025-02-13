import argparse
import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any

import numpy as np
from mlmodelscope import MLModelScope

# Constants
TRACE_LEVEL = (
    "NO_TRACE",
    "APPLICATION_TRACE",
    "MODEL_TRACE",              # pipelines within model
    "FRAMEWORK_TRACE",          # layers within framework
    "ML_LIBRARY_TRACE",         # cudnn, ...
    "SYSTEM_LIBRARY_TRACE",     # cupti
    "HARDWARE_TRACE",           # perf, papi, ...
    "FULL_TRACE"                # includes all of the above
)

def parse_args():
    parser = argparse.ArgumentParser(description="mlmodelscope")
    parser.add_argument("--standalone", type=str, nargs='?', default="true", choices=["false", "true"], help="Whether standalone(not connect with frontend)") 
    parser.add_argument("--agent", type=str, nargs='?', default="pytorch", choices=["pytorch", "tensorflow", "onnxruntime", "mxnet", "jax"], help="Which framework to use") 

    if parser.parse_known_args()[0].standalone == 'true': 
        parser.add_argument("--user", type=str, nargs='?', default="default", help="The name of the user") 
        parser.add_argument("--task", type=str, nargs='?', default="image_classification", help="The name of the task to predict.") 
        parser.add_argument("--model_name", type=str, nargs='?', default="torchvision_alexnet", help="The name of the model") 
        parser.add_argument("--config_file", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to use config file (.json)") 
        parser.add_argument("--config_file_path", type=str, nargs='?', default="config.json", help="The path of the config file") 
        parser.add_argument("--architecture", type=str, nargs='?', default="gpu", choices=["cpu", "gpu"], help="Which Processing Unit to use") 
        parser.add_argument("--num_warmup", type=int, nargs='?', default=2, help="Total number of warmup steps for predict.") 
        parser.add_argument("--dataset_name", type=str, nargs='?', default="test_data", help="The name of the dataset for predict.") 
        parser.add_argument("--batch_size", type=int, nargs='?', default=2, help="Total batch size for predict.") 
        parser.add_argument("--trace_level", type=str, nargs='?', default="NO_TRACE", choices=TRACE_LEVEL, help="MLModelScope Trace Level") 
        parser.add_argument("--gpu_trace", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to trace GPU activities") 
        parser.add_argument("--cuda_runtime_driver_time_adjustment", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to adjust the CUDA Runtime/Driver time")
        parser.add_argument("--save_trace_result", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to save the trace result")
        parser.add_argument("--save_trace_result_path", type=str, nargs='?', default="trace_result.txt", help="The path of the trace result file")
        parser.add_argument("--detailed_result", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to get detailed result") 
        parser.add_argument("--security_check", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to perform security check on the model file")
        parser.add_argument("--save_output", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to save the output")
        parser.add_argument("--save_output_path", type=str, nargs='?', default="output.json", help="The path of the output file")

    else: 
        parser.add_argument("--env_file", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to use env file") 
        
        if parser.parse_known_args()[0].env_file == 'false': 
            parser.add_argument("--db_dbname", type=str, nargs='?', required=True, help="The name of the database") 
            parser.add_argument("--db_host", type=str, nargs='?', default='localhost', help="The host of the database") 
            parser.add_argument("--db_port", type=int, nargs='?', default=15432, help="The port of the database") 
            parser.add_argument("--db_user", type=str, nargs='?', required=True, help="The user of the database") 
            parser.add_argument("--db_password", type=str, nargs='?', required=True, help="The password of the database") 
            
            parser.add_argument("--mq_name", type=str, nargs='?', required=True, help="The name of the messagequeue") 
            parser.add_argument("--mq_host", type=str, nargs='?', default='localhost', help="The user of the messagequeue") 
            parser.add_argument("--mq_port", type=int, nargs='?', default=5672, help="The port of the messagequeue") 
            parser.add_argument("--mq_user", type=str, nargs='?', required=True, help="The user of the messagequeue") 
            parser.add_argument("--mq_password", type=str, nargs='?', required=True, help="The password of the messagequeue") 

    return parser.parse_args()

class DatabaseConnection:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: int):
        self.conn = self._connect_to_db(host, dbname, user, password, port)
        self.cur = self.conn.cursor()

    @staticmethod
    def _connect_to_db(host: str, dbname: str, user: str, password: str, port: int):
        try:
            import psycopg
        except ImportError:
            raise ImportError("Please install psycopg to use database functionality")
        
        return psycopg.connect(
            f"host={host} dbname={dbname} user={user} password={password} port={port}"
        )

    def update_trial(self, trial_id: str, result: Dict[str, Any]) -> None:
        dt = datetime.now(timezone.utc)
        query = f"UPDATE trials SET updated_at = %s, completed_at = %s, result = %s WHERE id = %s;"
        try:
            # 'single quote' produces errors in frontend
            self.cur.execute(query, (dt, dt, str(json.dumps(result)), trial_id))
        except Exception as e:
            print(f"Database error: {e}")
            self.conn.rollback()
        else:
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

class MessageQueueHandler:
    def __init__(self, host: str, port: int, user: str, password: str, queue_name: str):
        self.connection, self.channel = self._connect_to_queue(host, port, user, password, queue_name)
        self.queue_name = queue_name

    @staticmethod
    def _connect_to_queue(host: str, port: int, user: str, password: str, queue_name: str):
        try:
            import pika
        except ImportError:
            raise ImportError("Please install pika to use message queue functionality")

        credentials = pika.PlainCredentials(user, password)
        parameters = pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)
        return connection, channel

    def start_consuming(self, callback) -> None:
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback,
            auto_ack=True
        )
        try:
            print("Waiting for messages. To exit press CTRL+C")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Exiting...")
            self.channel.stop_consuming()

    def close(self) -> None:
        self.connection.close()

def process_message(db_conn: DatabaseConnection, body: bytes, properties, agent: str) -> None:
    received_message = json.loads(body.decode())
    
    # Extract message parameters
    user = received_message.get('User', 'default')
    task = received_message['DesiredResultModality']
    architecture = 'gpu' if received_message['UseGpu'] else 'cpu'
    gpu_trace = received_message['UseGpu'] != "NO_TRACE"

    if architecture != "gpu" and gpu_trace:
        gpu_trace = False
        print("GPU trace disabled for CPU architecture")

    # model_name = received_message['ModelName'][:-4].lower().replace('.', '_')
    model_name = received_message['ModelName'].lower().replace('.', '_')
    num_warmup = received_message.get('NumWarmup', 0)
    dataset_name = received_message['InputFiles']
    batch_size = received_message.get('BatchSize', 1)

    config = received_message.get('Config', None)
    security_check = received_message.get('SecurityCheck', False)

    duration_start = time.time()
    mlms = MLModelScope(architecture, received_message.get('TraceLevel', 'NO_TRACE'), gpu_trace)

    mlms.load_agent(task, agent, model_name, security_check, config, user)
    print(f"{agent}-agent loaded with {model_name} model")
    mlms.load_dataset(dataset_name, batch_size, None, security_check)
    print(f"{dataset_name} dataset loaded")
    print("Prediction starts")

    duration_for_inference_start_time = time.time()
    outputs = mlms.predict(num_warmup, True)
    duration_for_inference_end_time = time.time()
    duration_for_inference = (duration_for_inference_end_time - duration_for_inference_start_time)
    print("Prediction done")

    mlms.Close()
    duration_end_time = time.time()

    duration = (duration_end_time - duration_start)

    # result = {'duration': duration, 'duration_for_inference': duration_for_inference, 'responses': outputs} 
    result = outputs[0] 
    result["duration"] = f"{duration:.10f}s" 
    result["duration_for_inference"] = f"{duration_for_inference:.10f}s" 
    result["responses"][0]["id"] = str(uuid4()) 

    db_conn.update_trial(properties.correlation_id, result)

def run_standalone(args):
    agent = args.agent 
    user = args.user 
    task = args.task 
    architecture = args.architecture 
    trace_level = args.trace_level
    gpu_trace = TRACE_LEVEL.index(trace_level) >= TRACE_LEVEL.index("SYSTEM_LIBRARY_TRACE") and args.gpu_trace == "true"
    if architecture != "gpu" and gpu_trace:
        gpu_trace = False
        print(f"GPU device will not be used because \"{architecture}\" is chosen for architecture.\nTherefore, gpu_trace option becomes off.")
    
    model_name = args.model_name 
    config = load_config(args.config_file, args.config_file_path)
    num_warmup = args.num_warmup 
    dataset_name = args.dataset_name 
    batch_size = args.batch_size 
    detailed = args.detailed_result == "true"
    security_check = args.security_check == "true"

    save_trace_result_path = args.save_trace_result_path if args.save_trace_result == "true" and trace_level != "NO_TRACE" else None
    save_output_path = args.save_output_path if args.save_output == "true" else None 

    mlms = MLModelScope(architecture, trace_level, gpu_trace, save_trace_result_path, args.cuda_runtime_driver_time_adjustment == "true") 
    
    mlms.load_agent(task, agent, model_name, security_check, config, user) 
    print(f"{agent}-agent is loaded with {model_name} model\n") 
    mlms.load_dataset(dataset_name, batch_size, None, security_check) 
    print(f"{dataset_name} dataset is loaded\n") 
    print(f"prediction starts") 
    outputs = mlms.predict(num_warmup) 
    print("prediction is done\n") 

    print_outputs(outputs, task, detailed, save_output_path)

    mlms.Close() 

def load_config(config_file_flag, config_file_path):
    if config_file_flag == "true":
        try: 
            with open(config_file_path, 'r') as f:
                config = json.load(f)
                print(f"config file {config_file_path} is loaded")
                return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"config file {config_file_path} is not loaded: {e}") 
    return None

def print_outputs(outputs, task, detailed, save_output_path):
    print("outputs are as follows:") 
    if task in ["image_classification", "sentiment_analysis"]: 
        if detailed: 
            print(outputs) 
        else: 
            print(np.argmax(outputs, axis=1)) 
    elif task == "image_object_detection": 
        print("image_object_detection") 
        print(len(outputs)) 
        for output in outputs: 
            print(f"{len(output[0][0])} {len(output[1][0])} {len(output[2][0])}") 
    elif task in ["image_semantic_segmentation", "depth_estimation"]: 
        for index, output in enumerate(outputs): 
            print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}") 
    elif task in ["image_enhancement", "image_synthesis"]: 
        for index, output in enumerate(outputs): 
            print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}, channel: {len(output[0][0])}") 
    elif task == "image_instance_segmentation": 
        for index, output in enumerate(outputs): 
            print(f"outputs[{index}] width: {len(output[0][0])}, height: {len(output[0][0][0])}")
            for i, label in enumerate(output[1]): 
                print(f"labels[{i}]: {label}", end=' ')
            print(f"\noutputs[{index}] masks unique number list: {np.unique(output[0]).tolist()}\n")
    elif task == "image_instance_segmentation_raw": 
        print(len(outputs)) 
        print(len(outputs[0])) 
    elif task in ["speech_synthesis", "audio_generation"]: 
        for index, output in enumerate(outputs): 
            print(f"outputs[{index}] length: {len(output)}") 
    else: 
        for index, output in enumerate(outputs): 
            print(f"outputs[{index}]: {output}") 
        if task in ["text_to_text"] and save_output_path:
            with open(save_output_path, 'w') as f: 
                json.dump(outputs, f, indent=4) 
            print(f"result is saved in {save_output_path}") 

def run_non_standalone(args):
    db_config, mq_config = load_configs(args)
    db_conn = DatabaseConnection(
        db_config['host'], db_config['name'],
        db_config['user'], db_config['password'],
        int(db_config['port'])
    )
    
    mq_handler = MessageQueueHandler(
        mq_config['host'], int(mq_config['port']),
        mq_config['user'], mq_config['password'],
        mq_config['name']
    )

    def callback(ch, method, properties, body):
        process_message(db_conn, body, properties, args.agent)

    try:
        mq_handler.start_consuming(callback)
    finally:
        db_conn.close()
        mq_handler.close()

def load_configs(args):
    if args.env_file == 'false':
        db_config = {
            'name': args.db_dbname,
            'host': args.db_host,
            'port': args.db_port,
            'user': args.db_user,
            'password': args.db_password
        }
        mq_config = {
            'name': args.mq_name,
            'host': args.mq_host,
            'port': args.mq_port,
            'user': args.mq_user,
            'password': args.mq_password
        }
    else:
        db_config = {
            'name': os.environ['DB_DBNAME'],
            'host': os.environ['DB_HOST'],
            'port': os.environ['DB_PORT'],
            'user': os.environ['DB_USER'],
            'password': os.environ['DB_PASSWORD']
        }
        mq_config = {
            'name': f'agent-{args.agent}-amd64',
            'host': os.environ['MQ_HOST'],
            'port': os.environ['MQ_PORT'],
            'user': os.environ['MQ_USER'],
            'password': os.environ['MQ_PASSWORD']
        }
    return db_config, mq_config

def main():
    args = parse_args()
    
    if args.standalone == 'true':
        run_standalone(args)
    else:
        run_non_standalone(args)

if __name__ == "__main__":
    main()
