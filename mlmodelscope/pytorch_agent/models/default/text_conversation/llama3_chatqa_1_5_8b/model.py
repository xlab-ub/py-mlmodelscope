from ....pytorch_abc import PyTorchAbstractClass 

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM 

class PyTorch_Transformers_Llama3_ChatQA_1_5_8B(PyTorchAbstractClass):
  def __init__(self, config=None):
    self.config = config if config else {}

    model_id = "nvidia/ChatQA-1.5-8B" 
    # self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left') 
    self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
    self.model = AutoModelForCausalLM.from_pretrained(model_id) 
    
    self.tokenizer.pad_token = self.tokenizer.eos_token 

    self.terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    self.max_new_tokens = self.config.get('max_new_tokens', 128) 
    self.messages = self.config.get('messages', []) 
    self.context = self.config.get('context', None) 
    self.RAG = self.config.get('RAG', None) 
    
    if (self.context is not None) and (self.RAG is not None): 
      from transformers import AutoModel 
      ## load retriever tokenizer and model
      self.retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
      self.query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
      self.context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')
      
      print(f"type(self.context): {type(self.context)}")
      print(f"len(self.context): {len(self.context)}")
      if type(self.context) != list:
        self.context = [self.context] 

      self.context_max_new_tokens = self.config.get('context_max_new_tokens', 512) 
      self.preprocess = self.preprocess_rag 

    # Prompt Format when context is available 
    # System: {System}

    # {Context}

    # User: {Question}

    # Assistant: {Response}

    # User: {Question}

    # Assistant:

    # Prompt Format when context is not available 
    # System: {System}

    # User: {Question}

    # Assistant: {Response}

    # User: {Question}

    # Assistant:
    
  def get_formatted_input(self, messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
      if item['role'] == "user":
        ## only apply this instruction for the first user turn
        item['content'] = instruction + " " + item['content']
        break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation if context else system + "\n\n" + conversation
    
    return formatted_input

  def preprocess(self, input_texts):
    formatted_input = self.get_formatted_input(self.messages + [{"role": "user", "content": input_texts[0]}], self.context)
    tokenized_prompt = self.tokenizer(self.tokenizer.bos_token + formatted_input, return_tensors="pt") 
    self.input_ids_shape = tokenized_prompt.input_ids.shape 
    return tokenized_prompt 

  def preprocess_rag(self, input_texts):
    ### running retrieval
    ## convert query into a format as follows:
    ## user: {user}\nagent: {agent}\nuser: {user}
    messages = self.messages + [{"role": "user", "content": input_texts[0]}] 
    formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()

    query_input = self.retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt').to(self.device) 
    ctx_input = self.retriever_tokenizer(self.context, padding=True, truncation=True, max_length=self.context_max_new_tokens, return_tensors='pt').to(self.device) 
    query_emb = self.query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    ## Compute similarity scores using dot product and rank the similarity
    similarities = query_emb @ ctx_emb.transpose(0, 1)
    ranked_results = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)

    ## get top-n chunks (n=5)
    retrieved_context = "\n\n".join(self.context[idx] for idx in ranked_results[0,:5])

    formatted_input = self.get_formatted_input(messages, retrieved_context) 
    tokenized_prompt = self.tokenizer(self.tokenizer.bos_token + formatted_input, return_tensors="pt") 
    self.input_ids_shape = tokenized_prompt.input_ids.shape
    return tokenized_prompt 
    
  def predict(self, model_input):
    return self.model.generate(
      input_ids=model_input.input_ids, 
      attention_mask=model_input.attention_mask,
      pad_token_id=self.tokenizer.eos_token_id, 
      max_new_tokens=self.max_new_tokens, 
      eos_token_id=self.terminators
    )

  def postprocess(self, model_output):
    response = model_output[:, self.input_ids_shape[-1]:]
    return [self.tokenizer.decode(response[0], skip_special_tokens=True)]

  def to(self, device):
    self.device = device 
    self.model.to(device) 
    if self.RAG:
        self.query_encoder.to(device)
        self.context_encoder.to(device)

  def eval(self):
    self.model.eval()
    if self.RAG:
      self.query_encoder.eval()
      self.context_encoder.eval()
