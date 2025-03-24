#%%
import os
import json
import random

import torch as t
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset

purple = '\033[95m'
blue = '\033[94m'
brown = '\033[33m'
cyan = '\033[96m'
lime = '\033[92m'
yellow = '\033[93m'
red = "\033[38;5;196m"
pink = "\033[38;5;206m"
orange = "\033[38;5;202m"
green = "\033[38;5;34m"
gray = "\033[38;5;8m"
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

logpath = os.path.dirname(os.path.abspath(__file__)) + '\\kissylogs.json'
logs = json.load(open(logpath, 'r'))
messages = logs['messages']

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

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
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
total_toks_target = 50_000_000

current_seq = ""
current_seq_len = 0
seqs = []
seq_tok_lens = []
seq_str_lens = []
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
        seqs.append(current_seq + "\n")
        seq_tok_lens.append(current_seq_len)
        seq_str_lens.append(len(current_seq))
        total_toks += current_seq_len
        print(f"{green}added sequence of {current_seq_len:,} tokens{endc}")
        print(f"{green}Total tokens: {total_toks:,}{endc}")
        current_seq = formatted
        current_seq_len = msg_len
    else:
        current_seq = formatted + "\n" + current_seq
        current_seq_len += msg_len
    if total_toks >= total_toks_target:
        print(f"{yellow}reached target of {total_toks_target:,} tokens{endc}")
        break

for seq in seqs:
    print(seq)
    samples.append({"text": seq})


random_order_samples = random.sample(samples, len(samples))
kissy = Dataset.from_list(random_order_samples)
print(f"{bold + green}created dataset of {len(kissy):,} samples, with an average sequence len of {sum(seq_tok_lens)/len(seq_tok_lens):.3f}{endc}")
print(f"average string length was {sum(seq_str_lens)/len(seq_str_lens):.3f}")

#%%
kissy.save_to_disk("kissy_dataset")
kissy.push_to_hub("eekay/kissy")
#%%

kissy = Dataset.load_from_disk("kissy_dataset")
kissy[66]
# %%
