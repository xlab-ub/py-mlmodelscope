from ....pytorch_abc import PyTorchAbstractClass 

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification 
from torch.nn.functional import softmax 

class PyTorch_Transformers_DistilBERT_Base_Uncased_Finetuned_SST_2_English(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english") 

    self.features = list(self.model.config.id2label.values())

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)

  def predict(self, model_input): 
    return self.model(**model_input).logits

  def postprocess(self, model_output):
    return softmax(model_output, dim=1).tolist() 
