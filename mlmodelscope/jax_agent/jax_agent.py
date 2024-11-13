import os 
import pathlib 
import logging 
from typing import Callable 

import flax 
import jax
import jax.numpy as jnp
import optax 
import flax.linen as nn 
from flax.training import train_state, checkpoints
 
from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class JAX_Agent:
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None):
        self.tracer = tracer
        self.span, self.ctx = self.tracer.start_span_from_context("jax-agent", context=context, trace_level="APPLICATION_TRACE") 
        self.architecture = architecture 
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

    def method_interceptor(self, next_fun, args, kwargs, context): 
        layer = type(next_fun.args[0]) 
        layer_name = '.'.join([layer.__module__, layer.__name__])

        prev_ctx = self.tracer.extract_context() 
        span, curr_ctx = self.tracer.start_span_from_context(layer_name, context=prev_ctx, trace_level="FRAMEWORK_TRACE")
        self.tracer.inject_context(curr_ctx) 

        output = next_fun(*args, **kwargs) 

        span.end() 
        self.tracer.inject_context(prev_ctx) 

        return output 

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
            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                with nn.intercept_methods(self.method_interceptor): 
                    model_output = self.model.predict(model_input)
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)

    def load_optimizer(self, optimizer_name, optimizer_config=None):
        try:
            optimizer_class = getattr(optax, optimizer_name.lower())
            self.optimizer = optimizer_class(**(optimizer_config or {})) 
        except AttributeError:
            # supported_optimizers = [name for name in dir(optax)]
            supported_optimizers = [
                'adabelief', 'adadelta', 'adan', 'adagrad', 'adafactor', 'adam', 'adamax', 'adamaxw', 'adamw', 'amsgrad', 
                'fromage', 'lamb', 'lars', 'lbfgs', 'lion', 'nadam', 'nadamw', 'noisy_sgd', 'novograd', 
                'optimistic_gradient_descent', 'optimistic_adam', 'polyak_sgd', 'radam', 'rmsprop', 'rprop', 
                'sgd', 'sign_sgd', 'sm3', 'yogi'
            ]
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not found. Supported optimizers: {supported_optimizers}") 
    
    def load_loss_function(self, loss_name, loss_config=None):
        try:
            self.loss_function = getattr(optax.losses, loss_name)
            self.loss_function_config = loss_config or {} 
        except AttributeError:
            supported_losses = [name for name in dir(optax.losses) if not name.startswith('_')]
            raise NotImplementedError(f"Loss '{loss_name}' not found. Supported losses: {supported_losses}")
    
    def create_train_state(self):
        class TrainState(train_state.TrainState):
            loss_function: Callable = flax.struct.field(pytree_node=False)
        
        def loss_fn(logits, labels):
            return self.loss_function(logits, labels, **self.loss_function_config).mean()
        
        return TrainState.create(
            apply_fn=self.model.model.__call__,
            params=self.model.model.params,
            tx=self.optimizer,
            loss_function=loss_fn
        )

    def train(self, num_epochs, num_batches, train_dataloader, val_dataloader, output_processor, save_trained_model_path=None):
        total_batches_processed = 0
        train_losses = []
        val_losses = []

        state = self.create_train_state() 

        with self.tracer.start_as_current_span_from_context(f'{self.model_name} training', context=self.ctx, trace_level="APPLICATION_TRACE") as training_span:
            for epoch in range(num_epochs):
                epoch_loss, total_batches_processed, state = self._train_epoch(epoch, num_batches, train_dataloader, total_batches_processed, state)
                val_loss = self._validate(epoch, val_dataloader, state)
                
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
                
                total_batches_processed += len(train_dataloader)
                if num_batches and total_batches_processed >= num_batches:
                    print(f"Total number of batches processed ({total_batches_processed}) reached or exceeded the limit ({num_batches}). Stopping training.")
                    break
        
        self.model.model.params = state.params

        if save_trained_model_path:
            checkpoints.save_checkpoint(save_trained_model_path, {'model': state}, step=epoch, overwrite=True)

        return train_losses, val_losses
    
    def _train_epoch(self, epoch, num_batches, train_dataloader, total_batches_processed, state):
        running_loss = 0.0
        batches_this_epoch = 0

        with self.tracer.start_as_current_span_from_context(f"Epoch {epoch}", trace_level="APPLICATION_TRACE"):
            for index, data_and_labels in enumerate(train_dataloader):
                if num_batches and total_batches_processed >= num_batches:
                    break

                data, labels = map(list, zip(*data_and_labels))
                labels = jnp.asarray(labels)

                loss, state = self._train_batch(data, labels, epoch, index, state)
                running_loss += loss
                batches_this_epoch += 1
                total_batches_processed += 1

                if (index + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    print(f"Epoch {epoch}, Batch {index + 1}, Loss: {avg_loss:.4f}")
                    running_loss = 0.0

        train_dataloader.reset()
        avg_epoch_loss = running_loss / (batches_this_epoch % 100 or 100)

        return avg_epoch_loss, total_batches_processed, state 

    def _train_batch(self, data, labels, epoch, index, state):        
        with self.tracer.start_as_current_span_from_context(f"Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

            def loss_function(params):
                with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                    self.tracer.inject_context(set_span_in_context(predict_span))
                    with nn.intercept_methods(self.method_interceptor):
                        logits = state.apply_fn(**model_input, params=params, train=True)[0]
                
                with self.tracer.start_as_current_span_from_context("compute_loss", trace_level="APPLICATION_TRACE"):
                    loss = state.loss_function(logits.logits, labels)
                return loss
        
            with self.tracer.start_as_current_span_from_context("compute_gradient", trace_level="APPLICATION_TRACE"):
                grad_function = jax.value_and_grad(loss_function) 
                loss, grad = grad_function(state.params) 
                # grad = jax.lax.pmean(grad, axis_name='batch') # mean the gradient across the batch on TPU
            
            with self.tracer.start_as_current_span_from_context("update_parameters", trace_level="APPLICATION_TRACE"):
                new_state = state.apply_gradients(grads=grad)
            
            return loss.item(), new_state

    def _validate(self, epoch, val_dataloader, state):
        running_val_loss = 0.0
        
        with self.tracer.start_as_current_span_from_context(f"Validation Epoch {epoch}", trace_level="APPLICATION_TRACE"):
            for index, data_and_labels in enumerate(val_dataloader):
                data, labels = map(list, zip(*data_and_labels))
                loss = self._validate_batch(data, labels, epoch, index, state)
                running_val_loss += loss

        val_dataloader.reset()
        avg_val_loss = running_val_loss / max(len(val_dataloader), 1)
        return avg_val_loss 
   
    def _validate_batch(self, data, labels, epoch, index, state):
        with self.tracer.start_as_current_span_from_context(f"Validate Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                with nn.intercept_methods(self.method_interceptor):
                    # model_output = self.model.predict(model_input)
                    model_output = state.apply_fn(**model_input, params=state.params, train=False)[0]

            labels = jnp.asarray(labels, dtype=jnp.int32) 
            with self.tracer.start_as_current_span_from_context("compute_loss", trace_level="APPLICATION_TRACE"):
                loss = self.loss_function(model_output, labels, **self.loss_function_config).mean()

        return loss.item()

    def Close(self):
        self.span.end()