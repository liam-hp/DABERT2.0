from torch import cuda
import torch.optim as optim
import torch.nn as nn
from transformers import BertConfig, PretrainedConfig
from datasets import load_dataset

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
dataset = load_dataset("embedding-data/simple-wiki", split="train")
sentences = [item for inner_list in dataset['set'] for item in inner_list]
trainSentences = sentences[:int(len(sentences) * 0.9)]
testSentences = sentences[int(len(sentences) * 0.9):]

device = "cuda" if cuda.is_available() else "cpu"
print(f"Setting device... {device}")

def train():
    
    epochs = hyperparams.get["epochs"]
    batch_size = hyperparams.get["batch_size"]
    max_sent_len = hyperparams.get["max_sentence_len"]
    test_epochs = hyperparams.get["test_epochs"]

    print("Initializing config, model, and optimizer...")
    config = PretrainedConfig.from_dict(copyHyperparams)

    model = architecture.CustomBertModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train_dataloader = batching.get_data_loader(trainSentences, batch_size=batch_size, max_length=max_sent_len)
    test_dataloader = batching.get_data_loader(trainSentences, batch_size=batch_size, max_length=max_sent_len)

    # training
    print(f"Beginning training on {epochs*batch_size} example sentences (approx. {round(epochs / len(train_dataloader), 2)}% of available)...")
    
    start_time = datetime.datetime.now()
    losses = []
    for epoch in range(epochs):
      optimizer.zero_grad()
      batch = next(iter(train_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
      if(100 * epoch / epochs % 5 == 0):
        print(f'\t {100 * epoch / epochs}% | Epoch {epoch} | Loss: {loss.item():.4f}', flush=True)
    print(f'\t 100% | Epoch {epoch} | Loss: {loss.item():.4f}', flush=True)
    
    training_time = datetime.timedelta(seconds=(datetime.datetime.now()-start_time).total_seconds())
    print(f"Training finished. Total training time (H:mm:ss): {training_time}")

    # validation
    avg_val_loss = 0
    print(f"Beginning validation on {test_epochs*batch_size} example sentences (approx. {round(test_epochs/len(test_dataloader), 2)}% of available)...")
    for epoch in range(test_epochs):
      batch = next(iter(test_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      avg_val_loss += loss.item()
    avg_val_loss /= test_epochs
    print(f"Validation complete. Avg validation loss: {avg_val_loss}")

    return 
    
if __name__ == '__main__': # This code won't run if this file is imported.
  train()