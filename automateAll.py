# from mlmodelscope.pytorch_agent.models.default.attention_detection.automation import (
#     attention_detection_model_automation,
# )
from mlmodelscope.pytorch_agent.models.default.document_visual_question_answering.automation import (
    document_visual_question_answering_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_enhancement.automation import (
    image_enhancement_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_synthesis.automation import (
    image_synthesis_model_automation,
)

# # from mlmodelscope.pytorch_agent.models.default.summarization.automation import (
# #     summarization_model_automation,
# # )

# from mlmodelscope.pytorch_agent.models.default.text_to_video.automation import (
#     text_to_video_model_automation,
# )
# Generalized modality-based imports
from mlmodelscope.pytorch_agent.models.default.audio_generation.automation import (
    text_to_audio_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_captioning.automation import (
    image_to_text_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_generation.automation import (
    text_to_image_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.automatic_speech_recognition.automation import (
    audio_to_text_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_classification.automation import (
    image_classification_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_object_detection.automation import (
    image_object_detection_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.sentiment_analysis.automation import (
    text_classification_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.text_to_code.automation import (
    text_to_code_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.video_classification.automation import (
    video_classification_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.depth_estimation.automation import (
    depth_estimation_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_editing.automation import (
    image_to_image_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.image_semantic_segmentation.automation import (
    image_semantic_segmentation_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.speech_synthesis.automation import (
    speech_synthesis_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.text_to_text.automation import (
    text_to_text_model_automation,
)
from mlmodelscope.pytorch_agent.models.default.visual_question_answering.automation import (
    visual_question_answering_model_automation,
)

import json

masterDataSet = json.load(open("categorized_huggingFaceModels.json"))


categories = {
    "Image Classification": [image_classification_model_automation],
    "Object detection": [image_object_detection_model_automation],
    "Instance segmentation": [image_semantic_segmentation_model_automation],
    "Style Transfer": [image_to_image_model_automation],
    # "Image to 3D": [image_to_3d_model_automation],
    "Mask Generation": [depth_estimation_model_automation],
    "Text to Image": [text_to_image_model_automation],
    # "Text to Video": [text_to_video_model_automation],
    "Image to Text": [image_to_text_model_automation],
    # "Text to 3D": [text_to_3d_model_automation],
    "Video Classification": [video_classification_model_automation],
    "Text to Text": [text_to_text_model_automation],
    "Text Classification": [text_classification_model_automation],
    "Text to Audio": [text_to_audio_model_automation],
    "Audio to Text": [audio_to_text_model_automation],
    # "Audio to Audio": [audio_to_audio_model_automation],
    # "Audio Classification": [audio_classification_model_automation],
    "Visual Question Answering": [visual_question_answering_model_automation],
    "Document Question Answering": [
        document_visual_question_answering_model_automation
    ],
    # "Table Editing": [table_editing_model_automation],
}

# [
# image_editing_model_automation -> image-to-image,
# sentiment_analysis_model_automation -> text-classification,
# image_generation_model_automation -> text-to-image,
# image_captioning_model_automation -> image-to-text,
# audio_generation_model_automation -> text-to-audio,
# automatic_speech_recognition_model_automation -> audio-to-text,

# ]


for modality in masterDataSet:
    if modality not in categories:
        continue
    print("Doing", modality)
    processorFunction = categories[modality][0]
    models = masterDataSet[modality]["allModels"]
    processorFunction(models)
