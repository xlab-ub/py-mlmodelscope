from ....pytorch_abc import PyTorchAbstractClass 
import requests
import torch
from PIL import Image

from transformers import SamModel, SamProcessor


# Generate segmentation masks given a 2D localization - 
# Generate segmentation masks per given localization (one prediction per 2D point)
# Generate segmentation masks given a bounding box
# Generate segmentation masks given a bounding box and a 2D points
# Generate segmentat

class Medsam_Vit_Base(PyTorchAbstractClass):
  
  def __init__(self, config=None):
    self.config = config if config else {}
    self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
    self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    self.input_points = [[[450, 600]]]
    self.sample_size = self.model.config.sample_size
    self.num_inference_steps = self.config.get('num_inference_steps', 25) # num_inference_steps: int = 1000 

  def preprocess(self, no_input): 
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    inputs = self.processor(raw_image, input_points=self.input_points, return_tensors="pt")
    return inputs
  
  def predict(self, model_input): 
    pv = model_input.pop("pixel_values", None)
    image_embeddings = self.model.get_image_embeddings(pv)
    model_input.update({"image_embeddings": image_embeddings})
    with torch.no_grad():
        outputs = self.model(**model_input)
    return outputs

  def postprocess(self, model_output):
    masks = self.processor.image_processor.post_process_masks(model_output.pred_masks.cpu(), model_output["original_sizes"].cpu(), model_output["reshaped_input_sizes"].cpu())
    scores = model_output.iou_scores
    return [masks,scores] 

  def to(self, device):
    self.device = device 
    self.processor.to(device)
    self.model.to(device) 