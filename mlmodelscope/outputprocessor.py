import base64
from io import BytesIO
from PIL import Image
import numpy as np

class OutputProcessor:
    # three types of processing outputs 
    # 1. processing post processed outputs 
    # method name : process_batch_outputs_postprocessed 
    # 2. processing the all outputs from the model to serialize the outputs 
    # method name : process_final_outputs_for_serialization
    # 3. processing the all outputs from the model for mlharness 
    # method name : process_final_outputs_for_mlharness

    outputs = [] 

    @classmethod
    def process_batch_outputs_postprocessed(cls, modality, batch_outputs):
        if modality in ["image_object_detection", "image_instance_segmentation", "image_instance_segmentation_raw"]: 
            for i in range(len(batch_outputs[0])): 
                cls.outputs.append([[output[i]] for output in batch_outputs]) 
        elif modality == "talking_head_generation":
            pass
        else: 
            cls.outputs.extend(batch_outputs) 
        
    @classmethod
    def get_final_outputs(cls): 
        # return the final outputs and reset the outputs
        final_outputs = cls.outputs
        cls.outputs = []
        return final_outputs 
    
    @staticmethod
    def process_final_outputs_for_serialization(modality, final_outputs, model_features=None): 
        if ((model_features is None) 
            and (modality not in ['image_enhancement', 'image_generation', 'image_synthesis', 'image_editing', 
                                  'speech_synthesis', 'audio_generation', 
                                  'text_to_code', 'text_to_text', 'automatic_speech_recognition', 'audio_to_text', 'visual_question_answering'])): 
            raise ValueError(f"model_features is required for {modality} modality") 
        
        serialized_outputs = []

        if modality in ['image_classification', 'sentiment_analysis', 'video_classification']: 
            for output in final_outputs: 
                features = [] 
                output = {k: v for k, v in enumerate(output)} 
                output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True)) 
                for idx, o in output.items(): 
                    features.append({"classification":{"index":idx,"label":model_features[idx]},"probability":round(o, 11),"type":"CLASSIFICATION"}) 
                    
                serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]}) 
        elif modality == 'image_object_detection': 
            for probabilities, classes, boxes in final_outputs: 
                features = [] 
                for p, c, b in zip(probabilities[0], classes[0], boxes[0]): 
                    features.append({"bounding_box":{"index":int(c),"label":model_features[c],"xmax":float(b[3]),"xmin":float(b[1]),"ymax":float(b[2]),"ymin":float(b[0])},"probability":round(float(p), 8),"type":"BOUNDINGBOX"}) 

                serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]}) 
        elif modality == 'image_semantic_segmentation': 
            for idx, output in enumerate(final_outputs): 
                features = [{"semantic_segment":{"height":len(output),"int_mask":[o_sub for o in output for o_sub in o],"labels":model_features,"width":len(output[0])},"probability":1,"type":"SEMANTICSEGMENT"}] 
            
                serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]})
        elif modality in ['image_enhancement', 'image_generation', 'image_synthesis', 'image_editing']: 
            for idx, output in enumerate(final_outputs): 
                img = Image.fromarray(np.array(output, dtype='uint8'), 'RGB') 
                buffer = BytesIO() 
                img.save(buffer, format="JPEG") 
                jpeg_data = base64.b64encode(buffer.getvalue()).decode('utf-8')  
                features = [{"raw_image":{"channels":len(output[0][0]),"char_list":None,"data_type":str(type(output[0][0][0])),"float_list":None,"height":len(output),"jpeg_data":jpeg_data,"width":len(output[0])},"probability":1,"type":"RAW_IMAGE"}] 

            serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]})
        elif modality in ['speech_synthesis', 'audio_generation']:
            for idx, output in enumerate(final_outputs): 
                audio_array = np.array(output) 
                channels = 1 if len(audio_array.shape) == 1 else audio_array.shape[0] 
                encoded_audio_data = base64.b64encode(audio_array.tobytes()).decode('utf-8')
                features = [{"audio":{"channels":channels,"data_type":str(audio_array.dtype),"raw_audio":encoded_audio_data,"sampling_rate":model_features["sampling_rate"]},"probability":1,"type":"RAW_AUDIO"}]

            serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]})
        
        elif modality in ['text_to_code', 'text_to_text', 'automatic_speech_recognition', 'audio_to_text', 'visual_question_answering']: 
            for idx, output in enumerate(final_outputs): 
                features = [{"text":output,"type":"TEXT"}] 

            serialized_outputs.append({"duration":None,"duration_for_inference":None,"responses":[{"features":features,"id":None}]})
        else: 
            raise NotImplementedError 
        return serialized_outputs 
    
    @staticmethod
    def process_final_outputs_for_mlharness(modality, final_outputs):
        mlharness_outputs = [] 
        if (modality == 'image_classification') or (modality == 'video_classification'):
            return np.argmax(final_outputs, axis=1) 
        elif modality == 'image_object_detection':
            for probabilities, classes, boxes in final_outputs: 
                features = [] 
                for p, c, b in zip(probabilities[0], classes[0], boxes[0]): 
                    features.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(p), float(c)]) 
                mlharness_outputs.append(features) 
        elif modality in ['question_answering', 'summarization']: 
            return final_outputs 
        else: 
            raise NotImplementedError 
        return mlharness_outputs 