#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import datasets
from datasets import Dataset

#%%

#model_name = "mistralai/Mistral-7B-v0.1"
model_name = "meta-llama/Llama-3.1-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for QLoRA

#%%
model.resize_token_embeddings(128257)
#adapter_name = "mistral_7b_lora_adapter"
adapter_name = "llama_8b_lora_adapter"
model = peft.PeftModel.from_pretrained(model, adapter_name)
model = model.merge_and_unload()
#model.save_pretrained("mistral_7b_kissy")
model.save_pretrained("llama_8b_kissy")
#%%