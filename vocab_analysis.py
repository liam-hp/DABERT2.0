from torch import cuda
import torch.optim as optim
import torch.nn as nn
from transformers import BertConfig, PretrainedConfig
from datasets import load_dataset
import random

# our files
import hyperparams
import batching
import architecture
import datetime

print("Fetching hyperparameters...")
copyHyperparams = {}
for k in hyperparams.get:
  if(hyperparams.get[k] != NotImplemented):
    copyHyperparams[k] = hyperparams.get[k]
    print(f"\t {k}: {hyperparams.get[k]}")

print("Loading in data...")
dataset = load_dataset("rahular/simple-wikipedia", split="train")
# print(dataset["text"])
# simple-wiki includes [complex, simple] pair of sentences. only use simple one
# sentences = [pair[0] for pair in dataset['text']]
sentences = dataset['text']

words = []
# words = [sen.split(" ") for sen in sentences]
for sen in sentences:
    wor = sen.split(" ")
    words += wor
# print(set(words))
# print(len(set(words)))
print(len(set(words)))
