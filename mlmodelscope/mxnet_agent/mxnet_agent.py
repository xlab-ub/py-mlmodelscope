import os 
import pathlib 
import logging 

from mxnet import nd, gluon, optimizer, autograd 

from opentelemetry.trace import set_span_in_context 

from ._load import _load 

logger = logging.getLogger(__name__) 

class MXNet_Agent: 
    def __init__(self, task, model_name, architecture, tracer, context, security_check=True, config=None, user='default', c=None): 
        self.tracer = tracer 
        self.c = c 

        self.span, self.ctx = self.tracer.start_span_from_context("mxnet-agent", context=context, trace_level="APPLICATION_TRACE") 

        self.architecture = architecture 

        self.load_model(task, model_name, security_check, config, user) 
  
    def load_model(self, task, model_name, security_check=True, config=None, user='default'):
        self.task = task
        model_path = f'{pathlib.Path(__file__).parent.resolve()}/models/{user}/{task}/{model_name}/model.py'
        if not os.path.exists(model_path):
            raise NotImplementedError(f"'{model_name}' model not found for '{task}' task and user '{user}'.")
        
        self.model_name = model_name
        with self.tracer.start_as_current_span_from_context(f'{self.model_name} model load', context=self.ctx, trace_level="APPLICATION_TRACE"):
            self.model = _load(task=task, model_name=self.model_name, architecture=self.architecture, security_check=security_check, config=config, user=user)

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
                if self.c:
                    self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                if self.c:
                    self.c.Close()
            with self.tracer.start_as_current_span_from_context("postprocess", trace_level="APPLICATION_TRACE"):
                return self.model.postprocess(model_output)
    
    def load_optimizer(self, optimizer_name, optimizer_config=None):
        try:
            self.optimizer = getattr(optimizer, optimizer_name)(**(optimizer_config or {}))
        except AttributeError:
            excluded_names = {'Optimizer', 'Test', 'Updater'}
            supported_optimizers = [name for name in dir(optimizer) if name[0].isupper() and name not in excluded_names]
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not found. Supported optimizers: {supported_optimizers}")

    def load_loss_function(self, loss_name, loss_config=None):
        try:
            self.loss_function = getattr(gluon.loss, loss_name)(**(loss_config or {}))
        except AttributeError:
            supported_losses = [name for name in dir(gluon.loss) if 'Loss' in name and name != 'Loss']
            raise NotImplementedError(f"Loss '{loss_name}' not found. Supported losses: {supported_losses}")

    def create_trainer(self):
        self.trainer = gluon.Trainer(self.model.model.collect_params(), self.optimizer)
    
    def train(self, num_epochs, num_batches, train_dataloader, val_dataloader, output_processor, save_trained_model_path=None):
        total_batches_processed = 0
        train_losses = []
        val_losses = []

        self.create_trainer()

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
        
        if save_trained_model_path:
            self.model.model.export(save_trained_model_path, epoch)
        
        return train_losses, val_losses

    def _train_epoch(self, epoch, num_batches, train_dataloader, total_batches_processed):
        running_loss = 0.0
        batches_this_epoch = 0

        with self.tracer.start_as_current_span_from_context(f"Epoch {epoch}", trace_level="APPLICATION_TRACE"):
            for index, data_and_labels in enumerate(train_dataloader):
                if num_batches and total_batches_processed >= num_batches:
                    break

                data, labels = map(list, zip(*data_and_labels))
                labels = nd.array(labels).as_in_context(self.model.ctx) 

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
            with autograd.record():
                model_output = self.model.predict(model_input)
            # if self.c:
            #     self.c.Close()
                with self.tracer.start_as_current_span_from_context("compute_loss", trace_level="APPLICATION_TRACE"): 
                    loss = self.loss_function(model_output, labels)

        with self.tracer.start_as_current_span_from_context("compute_gradient", trace_level="APPLICATION_TRACE"):
            loss.backward()
        
        with self.tracer.start_as_current_span_from_context("update_parameters", trace_level="APPLICATION_TRACE"):
            self.trainer.step(len(labels)) 

        return loss.mean().asscalar() 

    def _validate(self, epoch, val_dataloader):
        running_val_loss = 0.0
        
        with self.tracer.start_as_current_span_from_context(f"Validation Epoch {epoch}", trace_level="APPLICATION_TRACE"):
            for index, data_and_labels in enumerate(val_dataloader):
                data, labels = map(list, zip(*data_and_labels))
                labels = nd.array(labels).as_in_context(self.model.ctx)

                loss = self._validate_batch(data, labels, epoch, index)
                running_val_loss += loss

        val_dataloader.reset()
        avg_val_loss = running_val_loss / max(len(val_dataloader), 1)
        return avg_val_loss 
    
    def _validate_batch(self, data, labels, epoch, index):
        with self.tracer.start_as_current_span_from_context(f"Validate Batch {index}", trace_level="APPLICATION_TRACE"):
            with self.tracer.start_as_current_span_from_context("preprocess", trace_level="APPLICATION_TRACE"):
                model_input = self.model.preprocess(data)

            with self.tracer.start_as_current_span_from_context("predict", trace_level="MODEL_TRACE") as predict_span:
                self.tracer.inject_context(set_span_in_context(predict_span))
                # if self.c:
                #     self.c.Start(set_span_in_context(predict_span))
                model_output = self.model.predict(model_input)
                # if self.c:
                #     self.c.Close()
                with self.tracer.start_as_current_span_from_context("compute_loss", trace_level="APPLICATION_TRACE"):
                    loss = self.loss_function(model_output, labels).mean() 
        
        return loss.asscalar() 

    def Close(self):
        self.span.end()