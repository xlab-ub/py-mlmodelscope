import os 
import pathlib 
import logging 

# import json 
try:
    # import onnxruntime.training.onnxblock as onnxblock
    from onnxruntime.training.api import CheckpointState, Module, Optimizer
    from onnxruntime.training import artifacts
except ImportError:
    print("onnxruntime-training is not installed. Training functionality will not be available.")

from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class ONNXRuntime_Agent: 
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None): 
        self.tracer = tracer 
        self.c = c 

        # store the spans which are created for each prediction 
        # this is to find the corresponding context for the traced result
        self.spans_for_traced_result = {} 

        self.span, self.ctx = self.tracer.start_span_from_context("onnxruntime-agent", context=context, trace_level="APPLICATION_TRACE") 

        self.providers = ['CUDAExecutionProvider'] if architecture == "gpu" else ['CPUExecutionProvider'] 
        self.device = 'cuda' if architecture == "gpu" else 'cpu' 

        self.load_model(task, model_name, security_check, config, user) 

        if not self.tracer.is_trace_enabled("ML_LIBRARY_TRACE"): 
            self.model.disable_profiling()
        return 
  
    def load_model(self, task, model_name, security_check=True, config=None, user='default'): 
        self.task = task
        model_path = f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'
        if not os.path.exists(model_path):
            raise NotImplementedError(f"'{model_name}' model not found for '{task}' task and user '{user}'.")

        self.model_name = model_name
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} model load', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self.model = _load(task=task, model_name=self.model_name, providers=self.providers, security_check=security_check, config=config, user=user) 

    def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False):
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} start', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self._warmup(num_warmup, dataloader)
            final_outputs = self._evaluate(dataloader, output_processor)

        if serialized:
            return output_processor.process_final_outputs_for_serialization(self.task, final_outputs, getattr(self.model, 'features', None))
        elif mlharness:
            return output_processor.process_final_outputs_for_mlharness(self.task, final_outputs)
        return final_outputs

    def _warmup(self, num_warmup, dataloader):
        if num_warmup <= 0:
            return
        
        print('Warmup')
        num_warmup = min(num_warmup, len(dataloader))
        with self.tracer.start_as_current_span_from_context("Warmup", trace_level="APPLICATION_TRACE"):
            for index, data in enumerate(dataloader):
                if index >= num_warmup:
                    break
                self._process_batch(data, index, "Warmup")
        print('Warmup done')
        dataloader.reset()

    def _evaluate(self, dataloader, output_processor):
        with self.tracer.start_as_current_span_from_context("Evaluate", trace_level="APPLICATION_TRACE"):
            for index, data in enumerate(dataloader):
                post_processed_output = self._process_batch(data, index, "Evaluate")
                output_processor.process_batch_outputs_postprocessed(self.task, post_processed_output)
        return output_processor.get_final_outputs()

    def _process_batch(self, data, index, phase):
        with self.tracer.start_as_current_span_from_context(f"{phase} Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)
                if hasattr(model_input, 'to'):
                    model_input = model_input.to(self.device)
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                if self.c:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                if self.c:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)
  
    # # for self.spans_for_traced_result 
    # # all spans are ended, and we want to find the corresponding context for the traced result 
    # self.spans_for_traced_result = {k: v for k, v in sorted(self.spans_for_traced_result.items(), key=lambda item: item[1].start_time)} 

    # profiling_start_time = self.model.get_profiling_start_time_ns() 
    
    # profile_filename = self.model.get_profile_filename() 
    # with open(profile_filename) as f:
    #   traced_result = json.load(f) 
    # # print(len(traced_result)) 
    # # delete the profile file
    # os.remove(profile_filename) 

    # traced_result = sorted(traced_result, key=lambda event: event['ts']) 
    # # with open('_traced_result_onnx.json', 'w') as f: 
    # #   json.dump(traced_result, f, indent=4)
    
    # iter_parent_span = iter(self.spans_for_traced_result) 
    # layer_sequence_index = 0 
    # layerwise_prediction_results = {} 
    # layerwise_prediction_name = '' 
    # for event in traced_result:
    #   if (event['cat'] == "Session") and (event['name'] == "model_run"): 
    #     # Get the corresponding parent span for the traced result 
    #     layerwise_prediction_name = next(iter_parent_span) 
    #     layerwise_prediction_results[layerwise_prediction_name] = [] 
    #     parent_span = self.spans_for_traced_result[layerwise_prediction_name] 

    #     layer_sequence_index = 0 
    #   if (event['cat'] == "Node"): 
    #     event_start_time = (event['ts'] * 1000) + profiling_start_time 
    #     event_end_time = event_start_time + (event['dur'] * 1000) 

    #     event['args']['layer_sequence_index'] = layer_sequence_index 
    #     if event['name'].endswith('kernel_time'):
    #       layerwise_prediction_results[layerwise_prediction_name].append(event) 
    #     if event['name'].endswith('after'): 
    #       layer_sequence_index += 1 

    #     span, _ = self.tracer.start_span_from_context(name=event['name'], context=set_span_in_context(parent_span), trace_level="ML_LIBRARY_TRACE", attributes=event['args'], start_time=event_start_time) 
    #     span.end(end_time=event_end_time) 

    # top_row_num = 5 
    # for l_p_r in layerwise_prediction_results.keys():
    #   # print(f"len(layerwise_prediction_results[{l_p_r}]): {len(layerwise_prediction_results[l_p_r])}")
    #   layerwise_prediction_results[l_p_r] = sorted(layerwise_prediction_results[l_p_r], key=lambda event: event['dur'], reverse=True)
    #   layerwise_prediction_results[l_p_r] = layerwise_prediction_results[l_p_r][:top_row_num]
    #   print(f"The Top {top_row_num} most time consuming layers for {l_p_r}")
    #   print(f"{'Layer Index':<15}{'Layer Name':<50}{'Layer Type':<20}{'Latency(ms)':<20}")
    #   for event in layerwise_prediction_results[l_p_r]:
    #     print(f"{event['args']['layer_sequence_index']:<15}{event['name']:<50}{event['args']['op_name']:<20}{event['dur'] / 1000:<20.3f}")

    def load_optimizer(self, optimizer_name, optimizer_config=None):
        try:
            # self.optimizer_class = getattr(onnxblock.optim, optimizer_name)
            self.optimizer_class = getattr(artifacts.OptimType, optimizer_name)
        except AttributeError:
            # supported_optimizers = [name for name in dir(onnxblock.optim) if name[0].isupper()]
            supported_optimizers = [name for name in dir(artifacts.OptimType) if name[0].isupper()]
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not found. Supported optimizers: {supported_optimizers}")

    def load_loss_function(self, loss_name, loss_config=None):
        try:
            # self.loss_class = getattr(onnxblock.loss, loss_name)
            self.loss_class = getattr(artifacts.LossType, loss_name)
        except AttributeError:
            # supported_losses = [name for name in dir(onnxblock.loss) if 'Loss' in name]
            supported_losses = [name for name in dir(artifacts.LossType) if 'Loss' in name]
            raise NotImplementedError(f"Loss '{loss_name}' not found. Supported losses: {supported_losses}")

    def create_training_model_and_optimizer(self):
        def _get_grad_and_frozen_params(model):
            requires_grad = [] 
            frozen_params = [] 
            for param in model.graph.initializer:
                if 'running_mean' in param.name or 'running_var' in param.name:
                    frozen_params.append(param.name)
                else:
                    requires_grad.append(param.name)
            return requires_grad, frozen_params

        self.training_directory = f'{pathlib.Path(__file__).parent.resolve()}/tmp/{self.model_name}/training' 
        os.makedirs(self.training_directory, exist_ok=True)

        try:
            requires_grad, frozen_params = _get_grad_and_frozen_params(self.model.model)
            artifacts.generate_artifacts(
                self.model.model,
                optimizer=self.optimizer_class,
                loss=self.loss_class,
                requires_grad=requires_grad,
                frozen_params=frozen_params,
                artifact_directory=self.training_directory,
                additional_output_names=[output.name for output in self.model.model.graph.output]
            )
        except Exception as e:
            if 'domain_version' in e.__str__():
                from onnx import version_converter
                converted_model = version_converter.convert_version(self.model.model, 12)
                requires_grad, frozen_params = _get_grad_and_frozen_params(converted_model)
                artifacts.generate_artifacts(
                    converted_model,
                    optimizer=self.optimizer_class,
                    loss=self.loss_class,
                    requires_grad=requires_grad,
                    frozen_params=frozen_params,
                    artifact_directory=self.training_directory,
                    additional_output_names=[output.name for output in self.model.model.graph.output]
                )
            else:
                if os.path.exists(self.training_directory):
                    import shutil
                    shutil.rmtree(self.training_directory)
                raise e

        state = CheckpointState.load_checkpoint(f"{self.training_directory}/checkpoint")
        self.training_model = Module(f"{self.training_directory}/training_model.onnx", state, f"{self.training_directory}/eval_model.onnx")
        self.optimizer = Optimizer(f"{self.training_directory}/optimizer_model.onnx", self.training_model)

        # class LossBlock(onnxblock.TrainingBlock):
        #     def __init__(self, loss_class, loss_config):
        #         super(LossBlock, self).__init__()
        #         # self.loss = self.loss_class(**self.loss_config)
        #         print(f"loss_class: {loss_class}")
        #         self.loss = loss_class(**loss_config)
            
        #     def build(self, output_name):
        #         print(f"output_name: {output_name}")
        #         print(f"loss: {self.loss}")
        #         return self.loss(output_name), output_name

        # # training_block = LossBlock() 
        # training_block = LossBlock(self.loss_class, self.loss_config)
        # for param in self.model.model.graph.initializer:
        #     if 'running_mean' in param.name or 'running_var' in param.name:
        #         training_block.requires_grad(param.name, False)
        #     else:
        #         training_block.requires_grad(param.name, True)
        
        # # print(f"model output name: {self.model.output_name}")
        # model_params = None 
        # with onnxblock.base(self.model.model):
        #     # _ = training_block(*self.model.output_name)
        #     _ = training_block(*[output.name for output in self.model.model.graph.output])
        #     training_model, eval_model = training_block.to_model_proto() 
        #     model_params = training_block.parameters() 
        # print(f"ddone")
        
        # optimizer_block = self.optimizer_class() 
        # with onnxblock.empty_base() as accessor:
        #     _ = optimizer_block(model_params)
        #     optimizer_model = optimizer_block.to_model_proto() 

        # self.training_model = training_model
        # self.eval_model = eval_model
        # self.optimizer_model = optimizer_model
        # return training_model, eval_model, optimizer_model

    def train(self, num_epochs, num_batches, train_dataloader, val_dataloader, output_processor):
        total_batches_processed = 0
        train_losses = []
        val_losses = []

        # training_model, eval_model, optimizer_model = self.create_training_model_and_optimizer() 
        self.create_training_model_and_optimizer()

        with self.tracer.start_as_current_span_from_context(f'{self.model_name} training', context=self.ctx, trace_level="APPLICATION_TRACE") as training_span:
            for epoch in range(num_epochs):
                epoch_loss, total_batches_processed = self._train_epoch(epoch, num_batches, train_dataloader, total_batches_processed)
                val_loss = self._validate(epoch, val_dataloader)
                
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
                
                total_batches_processed += len(train_dataloader)
                if num_batches and total_batches_processed >= num_batches:
                    print(f"Total number of batches processed ({total_batches_processed}) reached or exceeded the limit ({num_batches}). Stopping training.")
                    break

        trained_model_path = f"{self.training_directory}/inference_model.onnx" 
        self.training_model.export_model_for_inferencing(trained_model_path, [output.name for output in self.model.model.graph.output])
        # TODO: Need to check if it works with the tasks such as style_transfer, text_to_text, etc.
        self.model.load_onnx(trained_model_path, self.providers, predict_method_replacement=False)

        return train_losses, val_losses

    def _train_epoch(self, epoch, num_batches, train_dataloader, total_batches_processed):
        running_loss = 0.0
        batches_this_epoch = 0

        self.training_model.train() 

        with self.tracer.start_as_current_span_from_context(f"Epoch {epoch}", trace_level="APPLICATION_TRACE"):
            for index, data_and_labels in enumerate(train_dataloader):
                if num_batches and total_batches_processed >= num_batches:
                    break

                data, labels = map(list, zip(*data_and_labels))
                loss = self._train_batch(data, labels, epoch, index)
                running_loss += loss
                batches_this_epoch += 1
                total_batches_processed += 1

                if (index + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    print(f"Epoch {epoch}, Batch {index + 1}, Loss: {avg_loss:.4f}")
                    running_loss = 0.0

        train_dataloader.reset()
        avg_epoch_loss = running_loss / (batches_this_epoch % 100 or 100)
        return avg_epoch_loss, total_batches_processed

    def _train_batch(self, data, labels, epoch, index):
        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

        with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
            self.tracer.inject_context(set_span_in_context(predict_span))
            # if self.c:
            #     self.c.Start(set_span_in_context(predict_span))
            train_loss, _ = self.training_model(model_input, labels)
            # if self.c:
            #     self.c.Close()

        with self.tracer.start_as_current_span_from_context("update_parameters", trace_level="APPLICATION_TRACE"):
            self.optimizer.step()
        
        self.training_model.lazy_reset_grad() 

        return train_loss 

    def _validate(self, epoch, val_dataloader):
        running_vloss = 0.0

        self.training_model.eval()

        with self.tracer.start_as_current_span_from_context("Validation", trace_level="APPLICATION_TRACE"):
            for index, data_and_vlabels in enumerate(val_dataloader):
                data, vlabels = map(list, zip(*data_and_vlabels))
                val_loss = self._validate_batch(data, vlabels, index)
                running_vloss += val_loss

        val_dataloader.reset()
        avg_val_loss = running_vloss / len(val_dataloader)
        return avg_val_loss
    
    def _validate_batch(self, data, labels, index):
        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

        with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
            self.tracer.inject_context(set_span_in_context(predict_span))
            # if self.c:
            #     self.c.Start(set_span_in_context(predict_span))
            val_loss, _ = self.training_model(model_input, labels)
            # if self.c:
            #     self.c.Close()

        return val_loss 

    def Close(self):
        self.span.end()