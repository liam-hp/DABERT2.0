# external
import random
import torch
from torch import cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig
from datasets import load_dataset

# our files
import hyperparams
import early_stopping
import batching
import architecture

device = "cuda" if cuda.is_available() else "cpu"

dataset = load_dataset("embedding-data/simple-wiki", split="train")
sentences = [item for inner_list in dataset['set'] for item in inner_list]
trainSentences = sentences[:int(len(sentences) * 0.9)]
testSentences = sentences[int(len(sentences) * 0.9):]

def train():
    print("Initializing...")

    config = BertConfig.from_pretrained('bert-base-uncased')
    model = architecture.CustomBertModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=8e-7)
    databatches = batching.get_data_loader(sentences)
    #early_stopping = early_stopping.EarlyStopping(tolerance=10, min_delta=0.01) if hyperparams["early_stopping"] else None
    losses = []

    print(f"Initialization complete. Beginning training on {len(trainSentences)} example sentences...")

    # training
    for batch_num, batch in enumerate(databatches):
      # batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()}  # move batch to the appropriate device
      outputs = model(**batch)  # unpack the batch dictionary directly into the model
      loss = outputs.loss  # the loss is returned when 'labels' are provided in the input
      losses.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(f'Batch Number: {batch_num} | Loss: {loss.item():.4f}')

    final_loss = sum(losses[-5:]) / 5
    return final_loss
    
if __name__ == '__main__': # This code won't run if this file is imported.
  train()