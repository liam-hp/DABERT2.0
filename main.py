from torch import cuda
import torch.optim as optim
import torch.nn as nn
from transformers import BertConfig
from datasets import load_dataset

# our files
import hyperparams
import batching
import architecture
import datetime

print("Fetching hyperparameters...")
for k in hyperparams.get:
  if(hyperparams.get[k] != NotImplemented):
    print(f"\t {k}: {hyperparams.get[k]}")

print("Loading in data...")
dataset = load_dataset("embedding-data/simple-wiki", split="train")
sentences = [item for inner_list in dataset['set'] for item in inner_list]
trainSentences = sentences[:int(len(sentences) * 0.9)]
testSentences = sentences[int(len(sentences) * 0.9):]

device = "cuda" if cuda.is_available() else "cpu"
print(f"Setting device... {device}")

def train():
    
    print("Initializing config, model, and optimizer...")
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = architecture.CustomBertModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train_dataloader = batching.get_data_loader(trainSentences, batch_size=hyperparams.get["batch_size"], max_length=hyperparams.get["max_sentence_len"])
    test_dataloader = batching.get_data_loader(trainSentences, batch_size=hyperparams.get["batch_size"], max_length=hyperparams.get["max_sentence_len"])

    # training
    training_size = hyperparams.get['epochs']*hyperparams.get['batch_size']
    print(f"Beginning training on {training_size} example sentences (approx. {round(hyperparams.get['epochs']/len(train_dataloader), 2)}% of available)...")
    
    start_time = datetime.datetime.now()
    losses = []
    for epoch in range(hyperparams.get["epochs"]):
      optimizer.zero_grad()
      batch = next(iter(train_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
      print(f'\t Epoch {epoch} | Loss: {loss.item():.4f}')
    
    training_time = datetime.timedelta((datetime.datetime.now()-start_time).total_seconds())
    print(f"Training finished. Total training time: {training_time}")

    # validation
    avg_val_loss = 0
    validation_size = hyperparams.get['test_epochs']*hyperparams.get['batch_size']
    print(f"Beginning validation on {validation_size} example sentences (approx. {round(hyperparams.get['test_epochs']/len(test_dataloader), 2)}% of available)...")
    for epoch in range(hyperparams.get["test_epochs"]):
      batch = next(iter(test_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      avg_val_loss += loss.item()
    avg_val_loss /= hyperparams.get['test_epochs']
    print(f"Validation complete. Avg validation loss: {avg_val_loss}")

    return 
    
if __name__ == '__main__': # This code won't run if this file is imported.
  train()