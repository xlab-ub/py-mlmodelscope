-- ==========================================
-- MLModelScope Text-to-Text Models Migration - COMPREHENSIVE
-- Migration: 002_add_all_text_to_text_models
-- ==========================================
-- Comprehensive migration including ALL text-to-text models from PyTorch, TensorFlow, and ONNX Runtime agents
-- Compatible with the UI task system (output_type = 'text_to_text')
-- Framework IDs: MXNet=1, ONNX Runtime=2, PyTorch=3, TensorFlow=4
-- Note: JAX agent exists but framework is not registered in database - skipping for now

BEGIN;

-- Insert ALL text-to-text models from supported agents
INSERT INTO models (
    name, short_description, description, framework_id, version, license,
    input_type, output_type, url_github, 
    created_at, updated_at
) VALUES 

-- ==========================================
-- PYTORCH AGENT MODELS (Framework ID = 3)
-- ==========================================

-- GPT Models (PyTorch)
(
    'gpt_2',
    'OpenAI GPT-2 - 117M parameter transformer model for text generation.',
    'OpenAI GPT-2 is a large transformer-based language model with 117M parameters, trained to predict the next word in a sequence. Capable of generating coherent text across various domains.',
    3, '1.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/openai/gpt-2', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'distilgpt2',
    'DistilGPT-2 - Distilled version of GPT-2 with 82M parameters.',
    'DistilGPT-2 is a distilled version of GPT-2 that retains 95% of the performance while being 33% smaller and 60% faster.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/huggingface/transformers', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- OPT Models (PyTorch)
(
    'opt_125m',
    'Meta OPT-125M - Open Pre-trained Transformer with 125M parameters.',
    'OPT-125M is part of Meta''s Open Pre-trained Transformer family, designed to be a fully open-source reproduction of GPT-3 with 125M parameters.',
    3, '1.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/facebookresearch/metaseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'opt_2_7b',
    'Meta OPT-2.7B - Open Pre-trained Transformer with 2.7B parameters.',
    'OPT-2.7B is a large-scale language model from Meta''s OPT family with 2.7B parameters, providing strong text generation capabilities.',
    3, '1.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/facebookresearch/metaseq', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- BLOOM Models (PyTorch)
(
    'bloom_560m',
    'BigScience BLOOM-560M - Multilingual autoregressive language model.',
    'BLOOM-560M is a multilingual large language model with 560M parameters, trained on 46 natural languages and 13 programming languages.',
    3, '1.0.0', 'BigScience RAIL License', 'text', 'text_to_text', 'https://github.com/bigscience-workshop/bigscience', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'bloomz_560m',
    'BigScience BLOOMZ-560M - Instruction-tuned multilingual model.',
    'BLOOMZ-560M is the instruction-tuned version of BLOOM-560M, fine-tuned on crosslingual task mixtures for improved zero-shot performance.',
    3, '1.0.0', 'BigScience RAIL License', 'text', 'text_to_text', 'https://github.com/bigscience-workshop/bigscience', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Llama Models (PyTorch)
(
    'llama_2_7b_hf',
    'Meta Llama 2 7B - Foundation model with 7B parameters.',
    'Llama 2 7B is a foundation language model from Meta, trained on 2 trillion tokens with improved performance over the original Llama.',
    3, '2.0.0', 'Llama 2 Community License', 'text', 'text_to_text', 'https://github.com/facebookresearch/llama', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_2_7b_chat_hf',
    'Meta Llama 2 7B Chat - Instruction-tuned conversational model.',
    'Llama 2 7B Chat is fine-tuned for dialog use cases with reinforcement learning from human feedback (RLHF) for safer, more helpful responses.',
    3, '2.0.0', 'Llama 2 Community License', 'text', 'text_to_text', 'https://github.com/facebookresearch/llama', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_2_13b_hf',
    'Meta Llama 2 13B - Larger foundation model with 13B parameters.',
    'Llama 2 13B is a larger foundation language model from Meta with 13B parameters, providing improved performance over the 7B variant.',
    3, '2.0.0', 'Llama 2 Community License', 'text', 'text_to_text', 'https://github.com/facebookresearch/llama', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_2_13b_chat_hf',
    'Meta Llama 2 13B Chat - Large instruction-tuned conversational model.',
    'Llama 2 13B Chat is the larger conversational variant fine-tuned for dialog with RLHF, providing enhanced conversational abilities.',
    3, '2.0.0', 'Llama 2 Community License', 'text', 'text_to_text', 'https://github.com/facebookresearch/llama', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'meta_llama_3_8b',
    'Meta Llama 3 8B - Next-generation foundation model.',
    'Llama 3 8B is Meta''s next-generation foundation model with improved architecture, training data, and performance across various tasks.',
    3, '3.0.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'meta_llama_3_8b_instruct',
    'Meta Llama 3 8B Instruct - Instruction-tuned variant.',
    'Llama 3 8B Instruct is fine-tuned for following instructions and conversational use cases with improved safety and helpfulness.',
    3, '3.0.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_3_2_1b',
    'Meta Llama 3.2 1B - Compact high-performance model.',
    'Llama 3.2 1B is a compact yet powerful language model designed for efficient deployment while maintaining strong performance.',
    3, '3.2.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_3_2_1b_instruct',
    'Meta Llama 3.2 1B Instruct - Compact instruction-tuned model.',
    'Llama 3.2 1B Instruct is the instruction-tuned version of the compact 1B model, optimized for following instructions efficiently.',
    3, '3.2.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_3_8b_instruct_gradient_1048k',
    'Meta Llama 3 8B Instruct Gradient - Extended context version.',
    'Llama 3 8B Instruct with extended context length of 1048k tokens using gradient-based techniques for long-form generation.',
    3, '3.0.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_3_refueled',
    'Llama 3 Refueled - Enhanced version with additional training.',
    'Llama 3 Refueled is an enhanced version with additional training data and fine-tuning for improved performance.',
    3, '3.0.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'hermes_2_pro_llama_3_8b',
    'Hermes 2 Pro Llama 3 8B - Advanced instruction-following model.',
    'Hermes 2 Pro is an advanced fine-tuned version of Llama 3 8B with enhanced instruction-following and reasoning capabilities.',
    3, '2.0.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/NousResearch/Hermes-2-Pro', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama3_chatqa_1_5_8b',
    'Llama3-ChatQA-1.5-8B - Question answering specialist model.',
    'Llama3-ChatQA-1.5-8B is specialized for conversational question answering with enhanced retrieval and reasoning capabilities.',
    3, '1.5.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/NVIDIA/ChatQA', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Gemma Models (PyTorch)
(
    'gemma_2b',
    'Google Gemma 2B - Lightweight high-performance model.',
    'Gemma 2B is Google''s lightweight language model with 2B parameters, designed for efficient deployment with strong performance.',
    3, '1.0.0', 'Gemma Terms of Use', 'text', 'text_to_text', 'https://github.com/google/gemma_pytorch', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'gemma_2b_it',
    'Google Gemma 2B IT - Instruction-tuned lightweight model.',
    'Gemma 2B IT is the instruction-tuned version of Gemma 2B, optimized for following instructions and conversational tasks.',
    3, '1.0.0', 'Gemma Terms of Use', 'text', 'text_to_text', 'https://github.com/google/gemma_pytorch', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'gemma_7b',
    'Google Gemma 7B - High-performance foundation model.',
    'Gemma 7B is Google''s foundation language model with 7B parameters, providing strong performance across various language tasks.',
    3, '1.0.0', 'Gemma Terms of Use', 'text', 'text_to_text', 'https://github.com/google/gemma_pytorch', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'gemma_7b_it',
    'Google Gemma 7B IT - Instruction-tuned foundation model.',
    'Gemma 7B IT is the instruction-tuned version of Gemma 7B, fine-tuned for better instruction following and safety.',
    3, '1.0.0', 'Gemma Terms of Use', 'text', 'text_to_text', 'https://github.com/google/gemma_pytorch', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Mistral Models (PyTorch)
(
    'mistral_7b_instruct_v0_1',
    'Mistral 7B Instruct v0.1 - Efficient instruction-following model.',
    'Mistral 7B Instruct v0.1 is an efficient instruction-following model with strong performance and fast inference.',
    3, '0.1.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/mistralai/mistral-src', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'mistral_7b_instruct_v0_2',
    'Mistral 7B Instruct v0.2 - Improved instruction-following model.',
    'Mistral 7B Instruct v0.2 is an improved version with better instruction following and enhanced safety features.',
    3, '0.2.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/mistralai/mistral-src', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'biomistral_7b',
    'BioMistral 7B - Medical domain specialized model.',
    'BioMistral 7B is a medical domain-specialized language model based on Mistral 7B, trained on biomedical literature.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/BioMistral/BioMistral', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Phi Models (PyTorch)
(
    'phi_3_mini_4k_instruct',
    'Microsoft Phi-3 Mini 4K Instruct - Compact high-quality model.',
    'Phi-3 Mini 4K Instruct is Microsoft''s compact yet high-quality language model optimized for instruction following with 4K context.',
    3, '3.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/microsoft/Phi-3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'phi_3_mini_128k_instruct',
    'Microsoft Phi-3 Mini 128K Instruct - Extended context model.',
    'Phi-3 Mini 128K Instruct extends the context length to 128K tokens while maintaining the compact and efficient design.',
    3, '3.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/microsoft/Phi-3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Qwen Models (PyTorch)
(
    'qwen2_5_0_5b_instruct',
    'Qwen2.5 0.5B Instruct - Ultra-compact multilingual model.',
    'Qwen2.5 0.5B Instruct is an ultra-compact multilingual model with strong performance despite its small size.',
    3, '2.5.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/QwenLM/Qwen2.5', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'qwen2_5_1_5b_instruct',
    'Qwen2.5 1.5B Instruct - Efficient multilingual model.',
    'Qwen2.5 1.5B Instruct provides efficient multilingual capabilities with enhanced instruction following.',
    3, '2.5.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/QwenLM/Qwen2.5', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Falcon Models (PyTorch)
(
    'falcon_7b_instruct',
    'Falcon 7B Instruct - High-performance instruction model.',
    'Falcon 7B Instruct is a high-performance instruction-following model with strong capabilities across various tasks.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/huggingface/transformers', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'falcon3_7b_instruct',
    'Falcon3 7B Instruct - Next-generation instruction model.',
    'Falcon3 7B Instruct is the next-generation version with improved architecture and training for better performance.',
    3, '3.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/huggingface/transformers', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- T5/Flan-T5 Models (PyTorch)
(
    'flan_t5_small',
    'Google Flan-T5 Small - Instruction-tuned T5 model.',
    'Flan-T5 Small is an instruction-tuned version of T5 with 80M parameters, trained on a collection of tasks with natural language instructions.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/google-research/t5x', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'flan_t5_base',
    'Google Flan-T5 Base - Balanced instruction-tuned model.',
    'Flan-T5 Base is a balanced instruction-tuned model with 250M parameters, providing good performance across various text tasks.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/google-research/t5x', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'flan_t5_large',
    'Google Flan-T5 Large - High-capacity instruction-tuned model.',
    'Flan-T5 Large is a high-capacity instruction-tuned model with 780M parameters for complex reasoning and generation tasks.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/google-research/t5x', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- UnifiedQA Models (PyTorch)
(
    'unifiedqa_t5_small',
    'UnifiedQA T5 Small - Multi-format question answering model.',
    'UnifiedQA T5 Small is trained on multiple QA datasets with different formats to handle diverse question-answering tasks.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/allenai/unifiedqa', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'unifiedqa_t5_large',
    'UnifiedQA T5 Large - Large-scale multi-format QA model.',
    'UnifiedQA T5 Large provides enhanced question-answering capabilities across multiple formats and domains.',
    3, '1.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/allenai/unifiedqa', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'unifiedqa_v2_t5_large_1363200',
    'UnifiedQA v2 T5 Large - Enhanced multi-format QA model.',
    'UnifiedQA v2 T5 Large is an enhanced version with improved training and better performance across various question-answering tasks.',
    3, '2.0.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/allenai/unifiedqa', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- DeepSeek Models (PyTorch)
(
    'deepseek_r1_distill_llama_8b',
    'DeepSeek R1 Distill Llama 8B - Distilled reasoning model.',
    'DeepSeek R1 Distill Llama 8B is a distilled model focused on reasoning tasks with enhanced logical capabilities.',
    3, '1.0.0', 'DeepSeek License', 'text', 'text_to_text', 'https://github.com/deepseek-ai/DeepSeek-R1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- ==========================================
-- TENSORFLOW AGENT MODELS (Framework ID = 4)
-- ==========================================

-- GPT Models (TensorFlow)
(
    'gpt_2',
    'TensorFlow GPT-2 - 117M parameter transformer for text generation.',
    'TensorFlow implementation of OpenAI GPT-2 with 117M parameters, optimized for TensorFlow ecosystem with efficient inference.',
    4, '1.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/openai/gpt-2', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- ==========================================
-- ONNX RUNTIME AGENT MODELS (Framework ID = 2)
-- ==========================================

-- BLOOM Models (ONNX Runtime)
(
    'bloom_560m',
    'ONNX BLOOM-560M - Optimized multilingual language model.',
    'ONNX Runtime implementation of BLOOM-560M with optimized inference for cross-platform deployment.',
    2, '1.0.0', 'BigScience RAIL License', 'text', 'text_to_text', 'https://github.com/bigscience-workshop/bigscience', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- GPT Models (ONNX Runtime)
(
    'gpt2',
    'ONNX GPT-2 - Optimized transformer for efficient inference.',
    'ONNX Runtime implementation of GPT-2 with optimized inference performance for production deployments.',
    2, '1.0.0', 'MIT', 'text', 'text_to_text', 'https://github.com/openai/gpt-2', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Llama Models (ONNX Runtime)
(
    'llama_3_2_1b',
    'ONNX Llama 3.2 1B - Compact optimized model.',
    'ONNX Runtime implementation of Llama 3.2 1B with optimized inference for efficient deployment.',
    2, '3.2.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),
(
    'llama_3_2_1b_instruct',
    'ONNX Llama 3.2 1B Instruct - Optimized instruction model.',
    'ONNX Runtime implementation of Llama 3.2 1B Instruct with optimized inference for instruction-following tasks.',
    2, '3.2.0', 'Llama 3 Community License', 'text', 'text_to_text', 'https://github.com/meta-llama/llama3', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
),

-- Qwen Models (ONNX Runtime)
(
    'qwen2_5_0_5b_instruct',
    'ONNX Qwen2.5 0.5B Instruct - Ultra-efficient multilingual model.',
    'ONNX Runtime implementation of Qwen2.5 0.5B Instruct with optimized inference for multilingual text generation.',
    2, '2.5.0', 'Apache-2.0', 'text', 'text_to_text', 'https://github.com/QwenLM/Qwen2.5', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
);

-- Verify the insertion
SELECT 'PyTorch Text-to-Text Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'text_to_text' AND framework_id = 3
UNION ALL
SELECT 'TensorFlow Text-to-Text Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'text_to_text' AND framework_id = 4
UNION ALL
SELECT 'ONNX Runtime Text-to-Text Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'text_to_text' AND framework_id = 2
UNION ALL
SELECT 'Total Text-to-Text Models' as category, COUNT(*) as model_count FROM models WHERE output_type = 'text_to_text';

-- Show framework breakdown
SELECT 
    f.name as framework,
    COUNT(*) as model_count
FROM models m
JOIN frameworks f ON m.framework_id = f.id
WHERE m.output_type = 'text_to_text'
GROUP BY f.name, f.id
ORDER BY f.id;

-- Show all model names grouped by framework
SELECT 
    f.name as framework,
    string_agg(m.name, ', ' ORDER BY m.name) as models
FROM models m
JOIN frameworks f ON m.framework_id = f.id
WHERE m.output_type = 'text_to_text'
GROUP BY f.name, f.id
ORDER BY f.id;

COMMIT;
