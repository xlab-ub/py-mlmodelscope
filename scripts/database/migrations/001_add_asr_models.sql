-- ==========================================
-- MLModelScope ASR Models Migration - COMPREHENSIVE
-- Migration: 001_add_all_asr_models
-- ==========================================
-- Comprehensive migration including ALL ASR models from PyTorch and TensorFlow agents
-- Compatible with the UI task system (output_type = 'audio_to_text')

BEGIN;

-- Insert ALL ASR models from PyTorch and TensorFlow agents
INSERT INTO models (
    name, short_description, description, framework_id, version, license,
    input_type, output_type, url_github, 
    created_at, updated_at
) VALUES 

-- ==========================================
-- PYTORCH AGENT MODELS (Framework ID = 3)
-- ==========================================

-- Whisper Models (PyTorch)
(
    'whisper_tiny_en',
    'OpenAI Whisper Tiny English model for automatic speech recognition with 39M parameters.',
    'OpenAI Whisper Tiny English is the smallest English-only variant with 39M parameters, designed for fast inference with acceptable accuracy for real-time applications.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_base_en',
    'OpenAI Whisper Base English model for automatic speech recognition with 74M parameters.',
    'OpenAI Whisper Base English is a Transformer-based ASR model trained on 680,000 hours of multilingual and multitask supervised data. This English-only variant provides good accuracy with reasonable computational requirements.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_small_en',
    'OpenAI Whisper Small English model for automatic speech recognition with 244M parameters.',
    'OpenAI Whisper Small English provides a balance between accuracy and speed with 244M parameters, suitable for applications requiring better accuracy than the tiny model.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_medium_en',
    'OpenAI Whisper Medium English model for automatic speech recognition with 769M parameters.',
    'OpenAI Whisper Medium English offers high accuracy with 769M parameters, providing excellent transcription quality for English audio content.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_tiny',
    'OpenAI Whisper Tiny multilingual model for automatic speech recognition with 39M parameters.',
    'OpenAI Whisper Tiny multilingual model supports 99+ languages with 39M parameters, designed for fast inference across multiple languages.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_base',
    'OpenAI Whisper Base multilingual model for automatic speech recognition with 74M parameters.',
    'OpenAI Whisper Base multilingual model supports 99+ languages with 74M parameters, providing good accuracy across multiple languages.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_small',
    'OpenAI Whisper Small multilingual model for automatic speech recognition with 244M parameters.',
    'OpenAI Whisper Small multilingual model supports 99+ languages with 244M parameters, offering improved accuracy over the base model.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_medium',
    'OpenAI Whisper Medium multilingual model for automatic speech recognition with 769M parameters.',
    'OpenAI Whisper Medium multilingual model supports 99+ languages with 769M parameters, providing high accuracy for diverse audio content.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large',
    'OpenAI Whisper Large multilingual model for automatic speech recognition with 1550M parameters.',
    'OpenAI Whisper Large is the most accurate multilingual model with 1550M parameters, providing state-of-the-art accuracy for transcription and translation tasks.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large_v2',
    'OpenAI Whisper Large V2 multilingual model with improved accuracy and robustness.',
    'OpenAI Whisper Large V2 is an improved version of the Large model with enhanced accuracy and robustness across languages and audio conditions.',
    3, '2.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large_v3',
    'OpenAI Whisper Large V3 - the latest and most advanced multilingual ASR model.',
    'OpenAI Whisper Large V3 is the latest iteration with improved performance on diverse audio conditions and better timestamp accuracy.',
    3, '3.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large_v3_turbo',
    'OpenAI Whisper Large V3 Turbo - optimized for faster inference.',
    'OpenAI Whisper Large V3 Turbo is optimized for faster inference while maintaining the accuracy of the Large V3 model.',
    3, '3.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Distil-Whisper Models (PyTorch)
(
    'distil_whisper_large_v2',
    'Distilled Whisper Large V2 - 6x faster than original with 99% performance.',
    'Distilled Whisper Large V2 is a compressed version that maintains 99% of the original performance while being 6x faster, ideal for real-time applications.',
    3, '2.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/huggingface/distil-whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'distil_whisper_large_v3',
    'Distilled Whisper Large V3 - 6x faster than original with 99% performance.',
    'Distilled Whisper Large V3 is the latest compressed version maintaining 99% performance while being significantly faster.',
    3, '3.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/huggingface/distil-whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'distil_whisper_medium_en',
    'Distilled Whisper Medium English - optimized for fast English transcription.',
    'Distilled Whisper Medium English is optimized for fast English transcription while maintaining high accuracy.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/huggingface/distil-whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'distil_whisper_small_en',
    'Distilled Whisper Small English - lightweight model for English transcription.',
    'Distilled Whisper Small English provides efficient English transcription with reduced computational requirements.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/huggingface/distil-whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Wav2Vec2 Models (PyTorch)
(
    'wav2vec2_base',
    'Facebook Wav2Vec2 Base model for automatic speech recognition.',
    'Wav2Vec2 Base is a self-supervised speech recognition model that learns representations from raw audio data.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'wav2vec2_base_960h',
    'Facebook Wav2Vec2 Base model fine-tuned on LibriSpeech 960h.',
    'Wav2Vec2 Base 960h is fine-tuned on 960 hours of LibriSpeech data, providing excellent English speech recognition.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'wav2vec2_large',
    'Facebook Wav2Vec2 Large model for automatic speech recognition.',
    'Wav2Vec2 Large is a larger self-supervised speech recognition model with improved accuracy over the base model.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'wav2vec2_large_xlsr_53_english',
    'Facebook Wav2Vec2 Large XLSR-53 model fine-tuned for English.',
    'Wav2Vec2 Large XLSR-53 English is a cross-lingual model fine-tuned specifically for high-quality English speech recognition.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'wav2vec2_conformer_rel_pos_large',
    'Facebook Wav2Vec2 Conformer with relative positional encoding.',
    'Wav2Vec2 Conformer with relative positional encoding combines Conformer architecture with Wav2Vec2 pretraining.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'wav2vec2_conformer_rope_large',
    'Facebook Wav2Vec2 Conformer with RoPE (Rotary Position Embedding).',
    'Wav2Vec2 Conformer with RoPE uses rotary position embeddings for improved positional understanding in speech recognition.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- HuBERT Models (PyTorch)
(
    'hubert_large_ls960_ft',
    'Facebook HuBERT Large model fine-tuned on LibriSpeech 960h.',
    'HuBERT Large LS960-FT uses self-supervised learning with masked prediction objectives, fine-tuned on LibriSpeech for excellent speech recognition.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'hubert_xlarge_ls960_ft',
    'Facebook HuBERT XLarge model fine-tuned on LibriSpeech 960h.',
    'HuBERT XLarge LS960-FT is the largest HuBERT model, providing state-of-the-art performance for speech recognition tasks.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- SeamlessM4T Models (PyTorch)
(
    'seamless_m4t_large',
    'Meta SeamlessM4T Large - multimodal and multilingual translation model.',
    'SeamlessM4T Large supports speech-to-speech, speech-to-text, text-to-speech, and text-to-text translation across 100+ languages.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/facebookresearch/seamless_communication', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'seamless_m4t_medium',
    'Meta SeamlessM4T Medium - efficient multimodal translation model.',
    'SeamlessM4T Medium provides efficient multimodal translation capabilities with reduced computational requirements.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/facebookresearch/seamless_communication', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'seamless_m4t_v2_large',
    'Meta SeamlessM4T V2 Large - improved multimodal translation model.',
    'SeamlessM4T V2 Large is an improved version with enhanced performance across all translation modalities.',
    3, '2.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/facebookresearch/seamless_communication', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- WhisperX Models (PyTorch)
(
    'whisperx_large_v2',
    'WhisperX Large V2 - Whisper with word-level timestamps and speaker diarization.',
    'WhisperX Large V2 combines Whisper transcription with accurate word-level timestamps and speaker diarization capabilities.',
    3, '2.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/m-bain/whisperX', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisperx_large_v3',
    'WhisperX Large V3 - latest Whisper with enhanced diarization.',
    'WhisperX Large V3 is the latest version with improved word-level timestamps and speaker diarization accuracy.',
    3, '3.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/m-bain/whisperX', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- CrisperWhisper Models (PyTorch)
(
    'crisperwhisper',
    'CrisperWhisper - optimized Whisper implementation for faster inference.',
    'CrisperWhisper is an optimized implementation of Whisper that provides faster inference while maintaining transcription quality.',
    3, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/nyrahealth/CrisperWhisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- ==========================================
-- TENSORFLOW AGENT MODELS (Framework ID = 4)
-- ==========================================

-- Whisper Models (TensorFlow)
(
    'whisper_tiny_en',
    'TensorFlow Whisper Tiny English model with 39M parameters.',
    'TensorFlow implementation of OpenAI Whisper Tiny English model, optimized for TensorFlow ecosystem with fast inference.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_base_en',
    'TensorFlow Whisper Base English model with 74M parameters.',
    'TensorFlow implementation of OpenAI Whisper Base English model, providing good accuracy with TensorFlow optimizations.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_small_en',
    'TensorFlow Whisper Small English model with 244M parameters.',
    'TensorFlow implementation of OpenAI Whisper Small English model, balancing accuracy and speed in TensorFlow.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_medium_en',
    'TensorFlow Whisper Medium English model with 769M parameters.',
    'TensorFlow implementation of OpenAI Whisper Medium English model, offering high accuracy for English transcription.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_tiny',
    'TensorFlow Whisper Tiny multilingual model with 39M parameters.',
    'TensorFlow implementation of OpenAI Whisper Tiny multilingual model, supporting 99+ languages with fast inference.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_base',
    'TensorFlow Whisper Base multilingual model with 74M parameters.',
    'TensorFlow implementation of OpenAI Whisper Base multilingual model, supporting 99+ languages with good accuracy.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_small',
    'TensorFlow Whisper Small multilingual model with 244M parameters.',
    'TensorFlow implementation of OpenAI Whisper Small multilingual model, providing improved accuracy across languages.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_medium',
    'TensorFlow Whisper Medium multilingual model with 769M parameters.',
    'TensorFlow implementation of OpenAI Whisper Medium multilingual model, offering high accuracy for diverse languages.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large',
    'TensorFlow Whisper Large multilingual model with 1550M parameters.',
    'TensorFlow implementation of OpenAI Whisper Large multilingual model, providing state-of-the-art accuracy.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'whisper_large_v2',
    'TensorFlow Whisper Large V2 multilingual model with improved performance.',
    'TensorFlow implementation of OpenAI Whisper Large V2 with enhanced accuracy and robustness.',
    4, '2.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/openai/whisper', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Wav2Vec2 Models (TensorFlow)
(
    'wav2vec2_base_960h',
    'TensorFlow Wav2Vec2 Base model fine-tuned on LibriSpeech 960h.',
    'TensorFlow implementation of Wav2Vec2 Base 960h, providing excellent English speech recognition with TensorFlow optimizations.',
    4, '1.0.0', 'MIT', 'audio', 'audio_to_text', 'https://github.com/pytorch/fairseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
);

-- Verify the insertion
SELECT 'PyTorch ASR Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'audio_to_text' AND framework_id = 3
UNION ALL
SELECT 'TensorFlow ASR Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'audio_to_text' AND framework_id = 4
UNION ALL
SELECT 'Total ASR Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'audio_to_text';

-- Show framework breakdown
SELECT 
    f.name as framework,
    COUNT(*) as model_count
FROM models m
JOIN frameworks f ON m.framework_id = f.id
WHERE m.output_type = 'audio_to_text'
GROUP BY f.name, f.id
ORDER BY f.id;

-- Show all model names grouped by framework
SELECT 
    f.name as framework,
    string_agg(m.name, ', ' ORDER BY m.name) as models
FROM models m
JOIN frameworks f ON m.framework_id = f.id
WHERE m.output_type = 'audio_to_text'
GROUP BY f.name, f.id
ORDER BY f.id;

COMMIT;
