#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import datasets
from datasets import Dataset

#%%

model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for QLoRA


def generate_text(model, tokenizer, prompt: str, num_new_tokens: int = 50) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=num_new_tokens,
        # Optionally, you can add parameters such as temperature=0.8, top_k=50, etc.
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text
print("Model and tokenizer loaded successfully.")
#%%
print("Generating text with the model...")
print(generate_text(model, tokenizer, "Once upon a time there was", num_new_tokens=100))  # Test the model

#%%


model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # for Mistral/LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

#%%

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # for TensorCore efficiency
)

#%%
kissy = Dataset.load_from_disk("kissy_dataset_tokenized")
print(kissy[0]["input_ids"])
print(tokenizer.decode(kissy[0]["input_ids"]))

#%%

training_args = TrainingArguments(
    output_dir="qlora-mistral-kissy",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",  # or "epoch" if you split kissy["validation"]
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    report_to="wandb",  # or "wandb" if logging to Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=kissy,
    data_collator=data_collator,
)

trainer.train()
#%%
model.save_pretrained("kissy-lora-only")        # LoRA adapter
merged = model.merge_and_unload()
merged.save_pretrained("kissy-merged-model")    # Full merged model
#%%