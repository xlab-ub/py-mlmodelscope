import subprocess 

import logging 
import argparse 

import numpy as np 

from mlmodelscope import MLModelScope 

logger = logging.getLogger(__name__) 

def main(): 
  parser = argparse.ArgumentParser(description="mlmodelscope") 
  parser.add_argument("--standalone", type=str, nargs='?', default="true", choices=["false", "true"], help="Whether standalone(not connect with frontend)") 
  parser.add_argument("--agent", type=str, nargs='?', default="pytorch", choices=["pytorch", "tensorflow", "onnxruntime", "mxnet"], help="Which framework to use") 

  if parser.parse_known_args()[0].standalone == 'true': 
    parser.add_argument("--task", type=str, nargs='?', default="eye_contact_detection", help="The name of the task to predict.") 
    parser.add_argument("--model_name", type=str, nargs='?', default="eye_contact_cnn", help="The name of the model") 
    parser.add_argument("--architecture", type=str, nargs='?', default="gpu", choices=["cpu", "gpu"], help="Which Processing Unit to use") 
    parser.add_argument("--num_warmup", type=int, nargs='?', default=0, help="Total number of warmup steps for predict.") 
    parser.add_argument("--dataset_name", type=str, nargs='?', default="eye_contact_detection data", help="The name of the dataset for predict.") 
    parser.add_argument("--batch_size", type=int, nargs='?', default=1, help="Total batch size for predict.") 
    parser.add_argument("--gpu_trace", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to trace GPU activities") 
    parser.add_argument("--detailed_result", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to get detailed result") 
    parser.add_argument("--security_check", type=str, nargs='?', default="false", choices=["false", "true"], help="Whether to perform security check on the model file")

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

  args = parser.parse_args() 
  
  agent = args.agent 
  
  if args.standalone == 'true': 
    task          = args.task 
    architecture  = args.architecture 
    gpu_trace     = True if args.gpu_trace == "true" else False 
    if architecture == "gpu": 
      # https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system 
      try:
        subprocess.check_output('nvidia-smi') 
        print('Nvidia GPU detected!') 
      except Exception as err: # this command not being found can raise quite a few different errors depending on the configuration
        raise RuntimeError('No Nvidia GPU in system!').with_traceback(err.__traceback__) 
    else: 
      if gpu_trace: 
        gpu_trace = False 
        print("GPU device will not be used because \"cpu\" is chosen for architecture.\nTherefore, gpu_trace option becomes off.") 

    model_name    = args.model_name 
    num_warmup    = args.num_warmup 
    dataset_name  = args.dataset_name 
    batch_size    = args.batch_size 
    detailed = True if args.detailed_result == "true" else False 
    security_check = True if args.security_check == "true" else False

    mlms = MLModelScope(architecture, gpu_trace) 
    
    mlms.load_agent(task, agent, model_name, security_check) 
    print(f"{agent}-agent is loaded with {model_name} model\n") 
    mlms.load_dataset(dataset_name, batch_size) 
    print(f"{dataset_name} dataset is loaded\n") 
    print(f"prediction starts") 
    outputs = mlms.predict(num_warmup, detailed) 
    print("prediction is done\n") 

    print("outputs are as follows:") 
    if task == "image_classification": 
      if detailed: 
        print(outputs) 
      else: 
          print(np.argmax(outputs, axis=1)) 
    elif task == "image_object_detection": 
      print("image_object_detection") 
      print(len(outputs)) 
      # print(len(outputs[0])) # 3 
      # for output in outputs: 
      #   print(output['boxes'].shape)
      #   print(output['scores'].shape)
      #   print(output['labels'].shape) 
      # print(outputs[0]) # boxes, scores, labels 
      for output in outputs: 
        # print(f"{len(output[0])} {len(output[1])} {len(output[2])}")
        print(f"{len(output[0][0])} {len(output[1][0])} {len(output[2][0])}") 
    elif task == "image_semantic_segmentation": 
      for index, output in enumerate(outputs): 
        print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}") 
    elif task == "image_enhancement": 
      for index, output in enumerate(outputs): 
        print(f"outputs[{index}] width: {len(output)}, height: {len(output[0])}, channel: {len(output[0][0])}") 
    elif task == "image_instance_segmentation": 
      # print(len(outputs)) 
      for index, output in enumerate(outputs): 
        # print(f"outputs[{index}] masks: {len(output[0])}, labels: {len(output[1])}")
        # print(f"masks[0]: {len(output[0][0])} masks[0][0]: {len(output[0][0][0])}")
        print(f"outputs[{index}] width: {len(output[0][0])}, height: {len(output[0][0][0])}")
        for i, label in enumerate(output[1]): 
          print(f"labels[{i}]: {label}", end=' ')
        print(f"\noutputs[{index}] masks unique number list: {np.unique(output[0]).tolist()}\n")
    elif task == "image_instance_segmentation_raw": 
      print(len(outputs)) 
      print(len(outputs[0])) # probs, labels, boxes, masks 
    else: 
      print(outputs) 

    mlms.Close() 
  
  else: # not standalone 
    import os 
    import time 
    from datetime import datetime, timezone 
    from uuid import uuid4 
    import json 
    
    import psycopg 
    import pika 

    detailed = True 

    if args.env_file == 'false': 
      db_name = args.db_dbname 
      db_host = args.db_host 
      db_port = args.db_port 
      db_user = args.db_user 
      db_password = args.db_password 

      mq_name = args.mq_name 
      mq_host = args.mq_host 
      mq_port = args.mq_port 
      mq_user = args.mq_user 
      mq_password = args.mq_password 
    else: 
      db_name = os.environ['DB_DBNAME'] 
      db_host = os.environ['DB_HOST'] 
      db_port = os.environ['DB_PORT'] 
      db_user = os.environ['DB_USER'] 
      db_password = os.environ['DB_PASSWORD'] 

      # mq_name = os.environ['mq_name'] 
      mq_name = f'agent-{agent}-amd64' 
      mq_host = os.environ['MQ_HOST'] 
      mq_port = os.environ['MQ_PORT'] 
      mq_user = os.environ['MQ_USER'] 
      mq_password = os.environ['MQ_PASSWORD']

    global conn 
    # Connect to an existing database 
    conn = psycopg.connect(f"host={db_host} dbname={db_name} user={db_user} password={db_password} port={db_port}") 
    
    global cur 
    # Open a cursor to perform database operations 
    cur = conn.cursor() 

    def callback(ch, method, properties, body):
      # This function will be called when a message is received from the queue
      global conn 
      global cur 

      id = properties.correlation_id 
      
      received_message = json.loads(body.decode()) 
      
      task          = received_message['DesiredResultModality'] 
      architecture  = 'gpu' if received_message['UseGpu'] else 'cpu' 
      gpu_trace     = True if received_message['UseGpu'] != "NO_TRACE" else False 
      if architecture == "gpu": 
        # https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system 
        try:
          subprocess.check_output('nvidia-smi') 
          print('Nvidia GPU detected!') 
        except Exception as err: # this command not being found can raise quite a few different errors depending on the configuration
          raise RuntimeError('No Nvidia GPU in system!').with_traceback(err.__traceback__) 
      else: 
        if gpu_trace: 
          gpu_trace = False 
          print("GPU device will not be used because \"cpu\" is chosen for architecture.\nTherefore, gpu_trace option becomes off.") 

      model_name    = received_message['ModelName'][:-4].lower().replace('.', '_') # _1.0 
      num_warmup    = received_message['NumWarmup'] 
      dataset_name  = received_message['InputFiles'] 
      batch_size    = received_message['BatchSize'] 

      # measure duration 
      duration_start_time = time.time()
      mlms = MLModelScope(architecture, gpu_trace) 
    
      mlms.load_agent(task, agent, model_name) 
      print(f"{agent}-agent is loaded with {model_name} model\n") 
      mlms.load_dataset(dataset_name, batch_size, task) 
      print(f"{dataset_name} dataset is loaded\n") 
      print(f"prediction starts") 
      # measure duration_for_inference 
      duration_for_inference_start_time = time.time()
      outputs = mlms.predict(num_warmup, detailed) 
      duration_for_inference_end_time = time.time()
      duration_for_inference = (duration_for_inference_end_time - duration_for_inference_start_time) 
      print(f"prediction done") 
      mlms.Close() 
      duration_end_time = time.time()

      duration = (duration_end_time - duration_start_time) 

      # result = {'duration': duration, 'duration_for_inference': duration_for_inference, 'responses': outputs} 
      result = outputs[0] 
      result["duration"] = f"{duration:.10f}s" 
      result["duration_for_inference"] = f"{duration_for_inference:.10f}s" 
      result["responses"][0]["id"] = str(uuid4()) 

      # https://www.geeksforgeeks.org/how-to-insert-current_timestamp-into-postgres-via-python/ 
      dt = datetime.now(timezone.utc) 
      
      try:
        query = f"UPDATE trials SET updated_at = %s, completed_at = %s, result = %s WHERE id = %s;"
        # https://stackoverflow.com/questions/18283725/how-to-create-a-python-dictionary-with-double-quotes-as-default-quote-format 
        # 'single quotes' produces errors in frontend 
        cur.execute(query, (dt, dt, str(json.dumps(result)), id)) 
      except BaseException as e:
        print(e) 
        conn.rollback()
      else:
        conn.commit()

    # Establish a connection to RabbitMQ server
    credentials = pika.PlainCredentials(mq_user, mq_password) 
    parameters = pika.ConnectionParameters(host=mq_host,
                                           port=mq_port,
                                           credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Declare a queue to consume messages from
    channel.queue_declare(queue=mq_name)

    # Set up the callback function to handle incoming messages
    channel.basic_consume(queue=mq_name, on_message_callback=callback, auto_ack=True)

    print(f"Waiting for messages. To exit press CTRL+C")

    # Start consuming messages from the queue
    try:
      channel.start_consuming()
    except KeyboardInterrupt:
      print("Exiting...")
      channel.stop_consuming()

    # Close the connection to database 
    conn.close() 

    # Close the connection to RabbitMQ
    connection.close()
    
    pass 

if __name__ == "__main__": 
  main() 