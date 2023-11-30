from ..tensorflow_abc import TensorFlowAbstractClass
import warnings
import tensorflow as tf
import numpy as np
import cv2

class Tensorflow_Efficientdet_d0(TensorFlowAbstractClass):

    def __init__(self):

        #Warning
        warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 

        #Load Model
        model_file_url = "efficientdet_d0" 
        model_path = self.model_file_download(model_file_url) 
        self.model = tf.saved_model.load(model_path)

        #Load Extra Params
        #self.target_image_size = (512, 512)

        #Load Features (COCO-2017)
        features_file_url = "https://s3.amazonaws.com/store.carml.org/synsets/coco/coco_labels_paper_background.txt" 
        self.features = self.features_download(features_file_url) 

    
    def preprocess(self, input_images):
        processed_images = []

        for image_path in input_images:
            # Model does not support batching. Model input should be of shape [1, height, width, 3]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = tf.cast(tf.expand_dims(img, axis = 0), dtype = tf.uint8)
            processed_images.append(processed_img)

        return processed_images
    

    def predict(self, model_input):
        output = []
        for image_tensor in model_input:
            output.append(self.model.signatures["serving_default"](image_tensor))

        return output
    

    #HARD NMS
    def postprocess(self, model_output, k=50, iou_threshold=0.5, score_threshold = 0.15):
        output = model_output[0]
        
        # Extract scores, boxes, and classes
        scores = tf.squeeze(output['detection_scores'], axis = 0).numpy()
        boxes = tf.squeeze(output['detection_boxes'], axis = 0).numpy()
        classes = tf.squeeze(output['detection_classes'], axis = 0).numpy()

        # Get the top k indices sorted by score
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Extract top k items
        top_k_scores = scores[top_k_indices]
        top_k_boxes = boxes[top_k_indices]
        top_k_classes = classes[top_k_indices]

        # Apply Non-Max Suppression
        selected_indices = tf.image.non_max_suppression(
            boxes=top_k_boxes,
            scores=top_k_scores,
            max_output_size=k,
            iou_threshold=iou_threshold,
            score_threshold = score_threshold,
         )
        
        # Create padded arrays
        padded_scores = np.ones_like(scores) * 1.0  # Initialize with 1.0
        padded_boxes = np.zeros_like(boxes)         # Initialize with zeros
        padded_classes = np.ones_like(classes) * 1.0  # Initialize with 1.0

        #Gather NMS outputs and directly update padded array with them
        for idx in selected_indices:
            original_idx = top_k_indices[idx]
            padded_scores[original_idx] = scores[original_idx]
            padded_boxes[original_idx] = boxes[original_idx]
            padded_classes[original_idx] = classes[original_idx]

        #Wrap the output to maintain form
        padded_scores = tf.expand_dims(padded_scores, axis = 0).numpy().tolist()
        padded_boxes = tf.expand_dims(padded_boxes, axis = 0).numpy().tolist()
        padded_classes = tf.expand_dims(padded_classes, axis = 0).numpy().tolist()


        return padded_scores, padded_classes, padded_boxes

        

