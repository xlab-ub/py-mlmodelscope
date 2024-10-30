import os
import pathlib
import logging
import types

import tensorflow as tf

from opentelemetry.trace import set_span_in_context
from ._load import _load

TF_VERSION = tf.__version__[0]
if TF_VERSION == '1':
    tf.compat.v1.enable_eager_execution()

logger = logging.getLogger(__name__)

class TensorFlow_Agent: 
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None):
        self.tracer = tracer
        self.all_spans = {}
        self.span, self.ctx = self.tracer.start_span_from_context(name="tensorflow-agent", context=context, trace_level="APPLICATION_TRACE")

        if architecture == "cpu":
            if TF_VERSION == '1':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            else:
                tf.config.set_visible_devices([], 'GPU')

        self.load_model(task, model_name, security_check, config, user)
        self.c = c

    def load_model(self, task, model_name, security_check=True, config=None, user='default'):
        self.task = task
        model_path = f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'
        if not os.path.exists(model_path):
            raise NotImplementedError(f"'{model_name}' model not found for '{task}' task and user '{user}'.")
        
        self.model_name = model_name
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} model load', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self.model = _load(task=task, model_name=self.model_name, security_check=security_check, config=config, user=user)
        
        if hasattr(self.model.model, "layers"):
            self._register_hooks(self.model.model.layers)

    def _register_hooks(self, layers):
        for layer in layers:
            layer._forward_pre_hook = self.pre_hook
            layer._forward_hook = self.hook
            layer._forward = layer.call
            layer.call = types.MethodType(self._proxy_call, layer)

    @staticmethod
    def _proxy_call(layer, *args, **kwargs):
        if layer._forward_pre_hook:
            layer._forward_pre_hook(layer, args, kwargs)
        output = layer._forward(*args, **kwargs)
        if layer._forward_hook:
            hook_result = layer._forward_hook(layer, output, args, kwargs)
            if hook_result is not None:
                output = hook_result
        return output

    def pre_hook(self, layer, *args, **kwargs):
        prev_ctx = self.tracer.extract_context()
        span, curr_ctx = self.tracer.start_span_from_context(layer.name, context=prev_ctx, trace_level="FRAMEWORK_TRACE")
        self.tracer.inject_context(curr_ctx)
        self.all_spans[layer.name] = (span, prev_ctx)

    def hook(self, layer, output, *args, **kwargs):
        span, prev_ctx = self.all_spans.pop(layer.name)
        span.end()
        self.tracer.inject_context(prev_ctx)

    def predict(self, num_warmup, dataloader, output_processor, serialized=False, mlharness=False):
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} start', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self._warmup(num_warmup, dataloader)
            final_outputs = self._evaluate(dataloader, output_processor)

        if serialized or mlharness:
            # EagerTensor is not JSON serializable
            final_outputs = [output.numpy().tolist() for output in final_outputs]

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
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)

    def load_optimizer(self, optimizer_name, optimizer_config=None):
        try:
            self.optimizer = getattr(tf.keras.optimizers, optimizer_name)(**(optimizer_config or {}))
        except AttributeError:
            supported_optimizers = [name for name in dir(tf.keras.optimizers) if name[0].isupper()]
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not found. Supported optimizers: {supported_optimizers}")

    def load_loss_function(self, loss_name, loss_config=None):
        try:
            self.loss_function = getattr(tf.keras.losses, loss_name)(from_logits=True, **(loss_config or {}))
        except AttributeError:
            supported_losses = [name for name in dir(tf.keras.losses) if not name.startswith('_')]
            raise NotImplementedError(f"Loss '{loss_name}' not found. Supported losses: {supported_losses}")

    def train(self, num_epochs, num_batches, train_dataloader, val_dataloader, output_processor):
        total_batches_processed = 0
        train_losses = []
        val_losses = []

        with self.tracer.start_as_current_span_from_context(f'{self.model_name} training', context=self.ctx, trace_level="APPLICATION_TRACE"):
            for epoch in range(num_epochs):
                epoch_loss, total_batches_processed = self._train_epoch(epoch, num_batches, train_dataloader, total_batches_processed)
                val_loss = self._validate(epoch, val_dataloader)
                
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
                
                if num_batches and total_batches_processed >= num_batches:
                    print(f"Total number of batches processed ({total_batches_processed}) reached or exceeded the limit ({num_batches}). Stopping training.")
                    break

        return train_losses, val_losses

    def _train_epoch(self, epoch, num_batches, train_dataloader, total_batches_processed):
        running_loss = 0.0
        batches_this_epoch = 0

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

            with tf.GradientTape() as tape:
                with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                    self.tracer.inject_context(set_span_in_context(predict_span))
                    if self.c:
                        self.c.Start(set_span_in_context(predict_span))
                    model_output = self.model.model(model_input, training=True)
                    if self.c:
                        self.c.Close()

                with self.tracer.start_as_current_span_from_context("calculate_loss", trace_level="APPLICATION_TRACE"):
                    loss = self.loss_function(labels, model_output)

            with self.tracer.start_as_current_span_from_context("compute_gradients", trace_level="APPLICATION_TRACE"):
                gradients = tape.gradient(loss, self.model.model.trainable_variables)

            with self.tracer.start_as_current_span_from_context("update_parameters", trace_level="APPLICATION_TRACE"):
                self.optimizer.apply_gradients(zip(gradients, self.model.model.trainable_variables))

        return loss.numpy()

    def _validate(self, epoch, val_dataloader):
        running_vloss = 0.0
        with self.tracer.start_as_current_span_from_context("Validation", trace_level="APPLICATION_TRACE"):
            for index, data_and_vlabels in enumerate(val_dataloader):
                data, vlabels = map(list, zip(*data_and_vlabels))
                val_loss = self._validate_batch(data, vlabels, index)
                running_vloss += val_loss

        val_dataloader.reset()
        avg_val_loss = running_vloss / len(val_dataloader)
        return avg_val_loss

    def _validate_batch(self, data, vlabels, index):
        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                if self.c:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.model(model_input, training=False)
                if self.c:
                    self.c.Close()

            with self.tracer.start_as_current_span_from_context("calculate_loss", trace_level="APPLICATION_TRACE"):
                val_loss = self.loss_function(vlabels, model_output)

        return val_loss.numpy()

    def Close(self):
        self.span.end()