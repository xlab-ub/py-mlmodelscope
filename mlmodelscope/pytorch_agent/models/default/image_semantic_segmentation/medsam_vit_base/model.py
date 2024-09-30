from ....pytorch_abc import PyTorchAbstractClass 
import torch
from PIL import Image

from transformers import SamModel, SamProcessor
class Medsam_Vit_Base(PyTorchAbstractClass):
  
  def __init__(self, config=None):
    self.config = config if config else {}
    self.model = SamModel.from_pretrained("facebook/sam-vit-huge")
    self.input_type = "2d_localization"  # 2d_localization, bounding_box, bounding_box_2d_points 
    self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    self.sample_size = self.model.config.sample_size
    self.num_inference_steps = self.config.get('num_inference_steps', 25) # num_inference_steps: int = 1000 

  def preprocess(self, inputs): 
    input_img, anchor = inputs[0], inputs[1]
    raw_image = Image.open(input_img).convert("RGB")
    if self.input_type == "2d_localization":
          inputs = self.processor(raw_image, input_points=anchor, return_tensors="pt")
    elif self.input_type == "bounding_box":
          inputs = self.processor(raw_image, input_box=[anchor], return_tensors="pt")
    elif self.input_type == "bounding_box_2d_points":
          inputs = self.processor(raw_image, input_box=[anchor], input_points=[anchor], return_tensors="pt")
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