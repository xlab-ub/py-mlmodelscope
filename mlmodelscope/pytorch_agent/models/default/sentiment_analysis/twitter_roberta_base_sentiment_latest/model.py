from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from torch.nn.functional import softmax

class PyTorch_Transformers_Twitter_RoBERTa_Base_Sentiment_Analysis(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    self.features = list(self.model.config.id2label.values())
  
  def preprocess_text(self, text):
    new_text = [] 
    for t in text.split(" "):
      t = '@user' if t.startswith('@') and len(t) > 1 else t
      t = 'http' if t.startswith('http') else t
      new_text.append(t)
    return " ".join(new_text)

  def preprocess(self, input_texts):
    for index, text in enumerate(input_texts):
      input_texts[index] = self.preprocess_text(text)
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)

  def predict(self, model_input): 
    return self.model(**model_input).logits

  def postprocess(self, model_output):
    return softmax(model_output, dim=1).tolist() 
