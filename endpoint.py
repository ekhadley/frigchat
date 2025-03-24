from transformers import AutoModelForCausalLM, AutoTokenizer
import runpod
import torch


model_path = "mistral_7b_kissy"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


def complete(model: AutoModelForCausalLM, tokenizer:AutoTokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    runpod.logging.info(f"Generating completion for prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    runpod.logging.info(f"input tokenized: {inputs['input_ids'].shape[1]} tokens. generating completion.")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    runpod.logging.info(f"generated completion.")
    comp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    runpod.logging.info(f"detokenized completion: '{comp}'")
    return comp

def handler(event):
    input = event['input']
    prompt = input.get('prompt')
    ntok = input.get('max_new_tokens', 50)
    comp = complete(model, tokenizer, prompt, max_new_tokens=ntok)
    return comp

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})