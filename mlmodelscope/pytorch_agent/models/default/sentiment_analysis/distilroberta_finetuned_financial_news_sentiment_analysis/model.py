from ....pytorch_abc import PyTorchAbstractClass 

from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from torch.nn.functional import softmax 

class PyTorch_Transformers_DistilRoBERTa_Finetuned_Financial_News_Sentiment_Analysis(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}
    self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    self.model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    self.features = list(self.model.config.id2label.values())

  def preprocess(self, input_texts):
    return self.tokenizer(input_texts, return_tensors="pt", padding=True)

  def predict(self, model_input): 
    return self.model(**model_input).logits

  def postprocess(self, model_output):
    return softmax(model_output, dim=1).tolist() 
