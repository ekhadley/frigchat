#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch

#%%

model_path = "mistral_7b_kissy"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistral_7b_lora_adapter")
model.eval()

def complete(prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


#%%
ds = datasets.Dataset.load_from_disk("./kissy_dataset")
#%%
idx = 104
msgs = ds[idx]['text']
#msgs = "\nXylotile: the thing i love most about sucking cocks is"

c = complete(msgs, max_new_tokens=250)

print(msgs)
print("====== MODEL CONTINUATION ========\n")
print(c)
#%%