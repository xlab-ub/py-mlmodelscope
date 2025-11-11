import os
import pathlib
import logging

import torch

from opentelemetry.trace import set_span_in_context

from ._load import _load

logger = logging.getLogger(__name__)

class PyTorch_Agent:
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None):
        self.tracer = tracer
        self.all_spans = {}
        self.span, self.ctx = self.tracer.start_span_from_context(name="pytorch-agent", context=context, trace_level="APPLICATION_TRACE")
        self.architecture = architecture
        if architecture == "gpu" and torch.cuda.is_available():
            self.device = 'cuda'
            self.multi_gpu = torch.cuda.device_count() > 1
        else:
            self.device = 'cpu'
            self.multi_gpu = False
        self.load_model(task, model_name, security_check, config, user)
        self.c = c

    def load_model(self, task, model_name, security_check=True, config=None, user='default'):
        self.task = task
        if task == 'audio_to_text':
            task = 'automatic_speech_recognition'

        model_path = f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'
        if not os.path.exists(model_path):
            raise NotImplementedError(f"'{model_name}' model not found for '{task}' task and user '{user}'.")
        
        self.model_name = model_name
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} model load', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self.model = _load(task=task, model_name=self.model_name, security_check=security_check, config=config, user=user, device=self.device, multi_gpu=self.multi_gpu)
            self.model.to(self.device, multi_gpu=self.multi_gpu)
            self.model.eval()

        if (
            hasattr(self.model, 'model') 
            and not any([
                hasattr(self.model.model, "isScriptModule"),
                'ScriptModule' in type(self.model.model).__name__
            ])
            and hasattr(self.model.model, "named_modules")
        ):
            self._register_hooks(self.model.model)

    def _register_hooks(self, model):
        self.input_shape = None
        counters = {}
        remove_list = []

        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.ModuleList):
                remove_list.append(name.split('.')[-1])
                continue

            revised_name = name
            for remove_layer in remove_list:
                if f".{remove_layer}." in name:
                    revised_name = name.replace(f".{remove_layer}.", ".")

            prefix = 'root' if revised_name == '' else 'root.'
            prefix_and_name = prefix + revised_name
            _prefix, counters = self._process_layer_name(prefix_and_name, counters, '.')

            layer_type_name = type(layer).__name__
            layer_name = f"{_prefix}__{name}__{layer_type_name}"

            layer.register_forward_pre_hook(self._pre_hook(layer_name))
            layer.register_forward_hook(self._hook(layer_name))

    def _process_layer_name(self, name, counters, separator='.'):
        parts = name.split(separator)
        if len(parts) == 1:
            if (0, parts[0]) not in counters:
                counters[(0, parts[0])] = 0
            return str(0), counters
        
        for i in range(len(parts)):
            key = (i, separator.join(parts[:i+1]))
            if key not in counters:
                counters[key] = 0

        if (len(parts) - 2, separator.join(parts[:-1])) in counters:
            if (len(parts) - 1, separator.join(parts[:-1])) not in counters:
                counters[(len(parts) - 1, separator.join(parts[:-1]))] = 0
            counters[(len(parts) - 1, separator.join(parts))] = counters[len(parts) - 1, separator.join(parts[:-1])]
            counters[len(parts) - 1, separator.join(parts[:-1])] += 1

        prefix = '-'.join(str(counters[(i, separator.join(parts[:i+1]))]) for i in range(len(parts)))
        return prefix, counters

    def _pre_hook(self, layer_name):
        def hook(module, input):
            prev_ctx = self.tracer.extract_context()
            span, curr_ctx = self.tracer.start_span_from_context(layer_name, context=prev_ctx, trace_level="FRAMEWORK_TRACE")
            self.tracer.inject_context(curr_ctx)
            self.all_spans[layer_name] = (span, prev_ctx)
            if layer_name.startswith('0-0__'):
                if input and hasattr(input[0], 'shape'):
                    self.input_shape = input[0].shape
        return hook

    def _hook(self, layer_name):
        def hook(module, input, output):
            span, prev_ctx = self.all_spans.pop(layer_name)
            span.set_attribute("layer_sequence_index", layer_name.split('__')[0])
            span.set_attribute("module", f"{type(module).__module__}.{type(module).__name__}")
            if input and hasattr(input[0], 'shape'):
                span.set_attribute("input_shape", str(input[0].shape))
            else:
                span.set_attribute("input_shape", str(self.input_shape) if layer_name.startswith('0__') else "None")
            
            # Handle output shape more robustly
            if hasattr(output, 'shape'):
                span.set_attribute("output_shape", str(output.shape))
            elif isinstance(output, (tuple, list)) and len(output) > 0 and hasattr(output[0], 'shape'):
                span.set_attribute("output_shape", str(output[0].shape))
            else:
                span.set_attribute("output_shape", "Unknown")
            
            span.end()
            self.tracer.inject_context(prev_ctx)
        return hook

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
                with torch.no_grad():
                    model_output = self.model.predict(model_input)
                if self.c:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)

    def load_optimizer(self, optimizer_name, optimizer_config=None):
        try:
            optimizer_class = getattr(torch.optim, optimizer_name)
            if optimizer_config and 'learning_rate' in optimizer_config:
                optimizer_config['lr'] = optimizer_config.pop('learning_rate')
            self.optimizer = optimizer_class(self.model.model.parameters(), **(optimizer_config or {}))
        except AttributeError:
            supported_optimizers = [name for name in dir(torch.optim) if name[0].isupper()]
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not found. Supported optimizers: {supported_optimizers}")

    def load_loss_function(self, loss_name, loss_config=None):
        try:
            loss_class = getattr(torch.nn, loss_name)
            self.loss_function = loss_class(**(loss_config or {}))
        except AttributeError:
            supported_losses = [name for name in dir(torch.nn) if 'Loss' in name]
            raise NotImplementedError(f"Loss '{loss_name}' not found. Supported losses: {supported_losses}")

    def train(self, num_epochs, num_batches, train_dataloader, val_dataloader, output_processor, save_trained_model_path=None):
        total_batches_processed = 0
        train_losses = []
        val_losses = []

        with self.tracer.start_as_current_span_from_context(f'{self.model_name} training', context=self.ctx, trace_level="APPLICATION_TRACE") as training_span:
            for epoch in range(num_epochs):
                epoch_loss = self._train_epoch(epoch, num_batches, train_dataloader, total_batches_processed)
                val_loss = self._validate(epoch, val_dataloader)
                
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
                
                total_batches_processed += len(train_dataloader)
                if num_batches and total_batches_processed >= num_batches:
                    print(f"Total number of batches processed ({total_batches_processed}) reached or exceeded the limit ({num_batches}). Stopping training.")
                    break
        
        if save_trained_model_path and hasattr(self.model.model, "named_modules"):
            for _, layer in self.model.model.named_modules():
                layer._forward_hooks.clear()
                layer._forward_pre_hooks.clear()

            if not save_trained_model_path.endswith('.pth'):
                save_trained_model_path += '.pth'
            torch.save(self.model.model, save_trained_model_path)

        return train_losses, val_losses

    def _train_epoch(self, epoch, num_batches, train_dataloader, total_batches_processed):
        self.model.model.train()
        running_loss = 0.0
        batches_this_epoch = 0

        with self.tracer.start_as_current_span_from_context(f"Epoch {epoch}", trace_level="APPLICATION_TRACE") as epoch_span:
            for index, data_and_labels in enumerate(train_dataloader):
                if num_batches and total_batches_processed + batches_this_epoch >= num_batches:
                    break

                data = [d for d, _ in data_and_labels]
                labels = torch.asarray([l for _, l in data_and_labels]).to(self.device)

                loss = self._train_batch(data, labels, epoch, index)
                running_loss += loss
                batches_this_epoch += 1

                if (index + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    print(f"Epoch {epoch}, Batch {index + 1}, Loss: {avg_loss:.4f}")
                    running_loss = 0.0
            
            train_dataloader.reset()

        avg_epoch_loss = running_loss / (batches_this_epoch % 100 or 100)
        return avg_epoch_loss

    def _train_batch(self, data, labels, epoch, index):
        self.optimizer.zero_grad()

        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE") as train_batch_span:
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as train_preprocess_span:
                model_input = self.model.preprocess(data)
                if hasattr(model_input, 'to'):
                    model_input = model_input.to(self.device)
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                if self.c is not None:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                if self.c is not None:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("calculate_loss", trace_level="APPLICATION_TRACE") as loss_span:
                loss = self.loss_function(model_output, labels)
            with self.tracer.start_as_current_span_from_context("compute_gradients", trace_level="APPLICATION_TRACE") as gradient_span:
                loss.backward()
            with self.tracer.start_as_current_span_from_context("update_parameters", trace_level="APPLICATION_TRACE") as update_span:
                self.optimizer.step()

        return loss.item()

    def _validate(self, epoch, val_dataloader):
        self.model.model.eval()
        running_vloss = 0.0

        with self.tracer.start_as_current_span_from_context("Validation", trace_level="APPLICATION_TRACE") as validation_span:
            with torch.no_grad():
                for index, data_and_vlabels in enumerate(val_dataloader):
                    data = [d for d, _ in data_and_vlabels]
                    vlabels = torch.asarray([l for _, l in data_and_vlabels]).to(self.device)

                    val_loss = self._validate_batch(data, vlabels, index)
                    running_vloss += val_loss
                
                val_dataloader.reset()

        avg_val_loss = running_vloss / len(val_dataloader)
        return avg_val_loss

    def _validate_batch(self, data, vlabels, index):
        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE") as val_batch_span:
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE") as val_preprocess_span:
                model_input = self.model.preprocess(data)
                if hasattr(model_input, 'to'):
                    model_input = model_input.to(self.device)
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                if self.c is not None:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                if self.c is not None:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("val_loss", trace_level="APPLICATION_TRACE") as val_loss_span:
                val_loss = self.loss_function(model_output, vlabels)

        return val_loss.item()

    def Close(self):
        self.span.end()