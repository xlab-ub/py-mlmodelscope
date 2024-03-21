from ....tensorflow_abc import TensorFlowAbstractClass 

import tensorflow as tf
import tensorflow_hub as hub

class TensorFlow_EfficientNet_V2_XL_ImageNet21K(TensorFlowAbstractClass):
  def __init__(self):
    self.img_size = (512, 512, 3) 
    self.model = tf.keras.Sequential([
      hub.KerasLayer("https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet21k-xl-classification/versions/1")
      ])
    self.model.build([None, self.img_size[0], self.img_size[1], self.img_size[2]]) 
    
    features_file_url = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt" 
    self.features = self.features_download(features_file_url) 

  # https://github.com/google/automl/blob/master/efficientnetv2/preprocessing.py#L58 
  def preprocess_for_eval(self, image, image_size, transformations=None):
    """Process image for eval."""
    transformations = transformations or ('crop' if image_size < 320 else '')
    if 'crop' in transformations:
      shape = tf.shape(image)
      height, width = shape[0], shape[1]
      ratio = image_size / (image_size + 32)  # for imagenet.
      crop_size = tf.cast(
          (ratio * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
      y, x = (height - crop_size) // 2, (width - crop_size) // 2
      image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, [image_size, image_size])

  def preprocess_image(self, image, image_size):
    image = self.preprocess_for_eval(image, image_size)
    image = (image - 128.0) / 128.0  # normalize to [-1, 1]
    return tf.cast(image, tf.float32)

  def preprocess(self, input_images):
    for i in range(len(input_images)):
      input_images[i] = self.preprocess_image(tf.image.decode_image(tf.io.read_file(input_images[i]), channels=3), self.img_size[0])
    model_input = tf.convert_to_tensor(input_images, dtype=tf.float32)

    return model_input
  
  def predict(self, model_input): 
    return self.model(model_input) 

  def postprocess(self, model_output):
    return tf.compat.v1.nn.softmax(model_output, dim=1) 
