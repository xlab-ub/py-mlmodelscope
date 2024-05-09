from ....pytorch_abc import PyTorchAbstractClass 

import re

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image 

class PyTorch_Transformers_Donut_Base_finetuned_DocVQA(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
    self.processor = DonutProcessor.from_pretrained(model_id) 
    self.model = VisionEncoderDecoderModel.from_pretrained(model_id) 

  def preprocess(self, input_document_images_and_questions): 
    images, questions = [], []
    for input_image, question in input_document_images_and_questions:
      images.append(Image.open(input_image).convert('RGB'))
      questions.append(f"<s_docvqa><s_question>{question}</s_question><s_answer>")
    return self.processor(images=images, text=questions, add_special_tokens=False, padding=True, return_tensors="pt") 
  
  def predict(self, model_input): 
    return self.model.generate(model_input['pixel_values'],
                                decoder_input_ids=model_input['labels'], 
                               max_length=self.model.decoder.config.max_position_embeddings,
                               pad_token_id=self.processor.tokenizer.pad_token_id,
                               eos_token_id=self.processor.tokenizer.eos_token_id,
                               use_cache=True,
                               bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                               return_dict_in_generate=True,
                               )

  def postprocess(self, model_output):
    sequences = self.processor.batch_decode(model_output.sequences) 
    answers = [] 
    for sequence in sequences:
      sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
      sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
      answer = self.processor.token2json(sequence)['answer']
      answers.append(answer)
    return answers
