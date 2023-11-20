from ..tensorflow_abc import TensorFlowAbstractClass
import warnings
import tensorflow as tf
import numpy as np
import cv2

class Tensorflow_Centernet_Resnet(TensorFlowAbstractClass):

    def __init__(self):

        #Warning
        warnings.warn("If the size of the images is not consistent, the batch size should be 1.") 

        #Load Model
        model_file_url = "centernet_resnet"
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
    

    def postprocess(self, model_output):
        detection_classes = model_output[0]['detection_classes'].numpy()
        detection_scores = model_output[0]['detection_scores'].numpy()
        detection_boxes = model_output[0]['detection_boxes'].numpy()

        probs, labels, boxes = [], [], []
        for i in range(len(detection_classes)):
            cur_probs, cur_labels, cur_boxes = [], [], []
            for j in range(len(detection_classes[i])):
                prob, label, box = detection_scores[i][j], detection_classes[i][j], detection_boxes[i][j].tolist()
                cur_probs.append(prob)
                cur_labels.append(label)
                cur_boxes.append(box)
            probs.append(cur_probs)
            labels.append(cur_labels)
            boxes.append(cur_boxes)

        print(labels)
        return probs, labels, boxes

    """
    def postprocess(self, model_output, k=50, iou_threshold=0.5):
        output = model_output[0]
        
        # Extract scores, boxes, and classes
        scores = np.squeeze(output['detection_scores'].numpy())
        boxes = np.squeeze(output['detection_boxes'].numpy())
        classes = np.squeeze(output['detection_classes'].numpy())

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
            iou_threshold=iou_threshold
         )

        # Select final information based on NMS indices
        nms_scores = tf.expand_dims(tf.gather(top_k_scores, selected_indices), axis = 0).numpy()
        nms_boxes = tf.expand_dims(tf.gather(top_k_boxes, selected_indices), axis = 0).numpy()
        nms_classes = tf.expand_dims(tf.gather(top_k_classes, selected_indices), axis = 0).numpy()

        print(nms_classes)
        return nms_scores, nms_classes, nms_boxes

        """

"""
    #not needed? Might be useful down the line
    def crop_and_resize(self, img):
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        target_height = tf.cast(self.target_image_size[0], dtype=tf.float32)
        target_width = tf.cast(self.target_image_size[1], dtype=tf.float32)

        y_scale = target_height/tf.cast(height, tf.float32)
        x_scale = target_width/tf.cast(width, tf.float32)
        min_scale = tf.minimum(y_scale, x_scale)

        self.image_scale = min_scale #Storing for reverting to original sizes

        scaled_y = height * min_scale
        scaled_x = width * min_scale

        scaled_img = tf.image.resize(img, [scaled_y, scaled_x], method = tf.image.ResizeMethod.BILINEAR)
        
        # Calculate padding offsets
        pad_height = (target_height - scaled_y) // 2
        pad_width = (target_width - scaled_x) // 2
        pad_height = tf.cast(pad_height, tf.int32)
        pad_width = tf.cast(pad_width, tf.int32)

        scaled_img = tf.image.pad_to_bounding_box(
            scaled_img,
            pad_height,
            pad_width,
            target_height,
            target_width
        )

        return scaled_img #Tensor w/ Dtype tf.float32
    def preprocess_image(self, img, dims=None, need_transpose=False):
        #Convert image to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.constant(img, dtype= tf.float32)

        #Normalize the image. Numbers are obtained from Google Brain Automl repo. (The default image normalization is identical to Cloud TPU ResNet).
        mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        

        img -= tf.constant(mean_rgb, shape=(1,1,3), dtype = tf.float32) #The shape ensures that the mean_rgb vector is the right shape for broadcasting
        img /= tf.constant(stddev_rgb, shape=(1,1,3), dtype = tf.float32)
        
        #img = tf.cast(self.crop_and_resize(img), dtype = tf.uint8)

        #image_scale = 1.0 / self.image_scale
        

        if need_transpose:
            img = tf.transpose(img, perm = [2, 0, 1])
        return img  
    """