#%%
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset
import torch as t
import os
import json

from utils import *
#%%

logpath = os.path.dirname(os.path.abspath(__file__)) + '\\kissylogs.json'
logs = json.load(open(logpath, 'r'))
messages = logs['messages']

#%%

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for QLoRA

#%%

print(f"keys of logs: {logs.keys()}")
print(f"keys of messages: {messages[0].keys()}")
print(f"number of messages: {logs['messageCount']:,}")
print(messages[-1])
#%%

def format(msg, timestamp=False):
    if isinstance(msg, list): return "".join([format(m, timestamp=timestamp)+"\n" for m in msg])
    if timestamp: return f"[{msg['timestamp'][:16].replace('T', ' ')}] {msg['author']['nickname']}: {msg['content']}"
    return f"{msg['author']['nickname']}: {msg['content']}"

def countToks(msg: str):
    return len(tokenizer.encode(msg))

#%%

last_10_msgs = messages[-10:]
print(format(last_10_msgs))

#%%

samples = []
kissy = Dataset.from_list(samples)

#%%

target_seq_len = 1024
total_toks_target = 5_000_000

current_seq = ""
current_seq_len = 0
seqs = []
seq_lens = []
total_toks = 0
msg_idx = len(messages)
while 1:
    msg_idx -= 1
    if msg_idx < 0:
        print(f"{red}ran out of messages!{endc}")
        break
    msg = messages[msg_idx]
    formatted = format(msg)
    print(formatted)
    msg_len = countToks(formatted)
    if msg_len > target_seq_len:
        print(f"{red}Message too long: {msg_len:,} tokens{endc}")
        print()
        continue
    if current_seq_len + msg_len > target_seq_len:
        seqs.append(current_seq)
        seq_lens.append(current_seq_len)
        total_toks += current_seq_len
        print(f"{green}added sequence of {current_seq_len:,} tokens{endc}")
        print(f"{green}Total tokens: {total_toks:,}{endc}")
        current_seq = formatted
        current_seq_len = msg_len
    else:
        current_seq = formatted + current_seq
        current_seq_len += msg_len
    if total_toks >= total_toks_target:
        print(f"{yellow}reached target of {total_toks_target:,} tokens{endc}")
        break

for seq in seqs:
    print(seq)
    samples.append({"text": seq})

kissy = Dataset.from_list(samples)
print(f"{bold + green}created dataset of {len(kissy):,} samples, with an average sequence len of {sum(seq_lens)/len(seq_lens):.3f}{endc}")

#%%
# Save the dataset
kissy.save_to_disk("kissy_dataset")

#%%

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding=False  # We'll use dynamic padding later
    )

kissy_tokenized = kissy.map(tokenize_fn, batched=True)
kissy_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
kissy_tokenized.save_to_disk("kissy_dataset_tokenized")

#%%