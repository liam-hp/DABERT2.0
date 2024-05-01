from torch import cuda
import torch.optim as optim
import torch.nn as nn
from transformers import BertConfig, PretrainedConfig
from datasets import load_dataset
import random
import os.path
import torch
from datetime import date

# our files
import hyperparams
import batching
import architecture
import datetime
import custom_print

custom_print.cprint("Fetching hyperparameters...", 'setting')
copyHyperparams = {}
for k in hyperparams.get:
  if(hyperparams.get[k] != NotImplemented):
    copyHyperparams[k] = hyperparams.get[k]
    print(f"\t {k}: {hyperparams.get[k]}")

custom_print.cprint("Loading in data...", 'setting')
dataset = load_dataset("embedding-data/simple-wiki", split="train")
# simple-wiki includes [complex, simple] pair of sentences. only use simple one
sentences = [pair[1] for pair in dataset['set']]
# random.seed(420) # seed for testing if we want
random.shuffle(sentences) # shuffle the data
trainSentences = sentences[:int(len(sentences) * 0.85)]
testSentences = sentences[int(len(sentences) * 0.85):]

device = "cuda" if cuda.is_available() else "cpu"
custom_print.cprint(f"Setting device... {device}", 'setting')

def train():
    
    epochs = hyperparams.get["epochs"]
    batch_size = hyperparams.get["batch_size"]
    max_sent_len = hyperparams.get["max_sentence_len"]
    test_epochs = hyperparams.get["test_epochs"]
    learning_rate = hyperparams.get["learning_rate"]

    custom_print.cprint("Initializing config, model, and optimizer...", "setting")
    config = PretrainedConfig.from_dict(copyHyperparams)

    model = architecture.CustomBertModel(config).to(device)

    # load in pretrained weights
    if(hyperparams.get["load_model_weights"]):
      path = hyperparams.get["load_weights_path"]
      if(os.path.exists(f"./saved_models/{path}.pt")): # if the path exists
        model_state_dict = model.state_dict() # save the initialized state dict of our model
        loaded_sd = torch.load(f'./saved_models/{path}.pt') # load in the saved dict of a previous model
        # keep only the loaded_sd keys that match keys in init_state_dict
        filtered_sd = {
          k: v for k, v in loaded_sd.items() if k in model_state_dict and 
                                                v.size() == model_state_dict[k].size()
        } 
        model_state_dict.update(filtered_sd)  # update existing layers in the init dict with the loaded dict
        model.load_state_dict(model_state_dict) # load the modified state_dict; uninitialized weights remain randomly initialized
      else:
        custom_print.cprint("Failed to load model weights", "save")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataloader = batching.get_data_loader(trainSentences, batch_size=batch_size, max_length=max_sent_len)
    test_dataloader = batching.get_data_loader(trainSentences, batch_size=batch_size, max_length=max_sent_len) # we should split the training data into train 80% validation 5/10% and rest testing

    # training
    custom_print.cprint(f"Beginning training on {epochs*batch_size} example sentences (approx. {round(epochs / len(train_dataloader), 2)}% of available)...", 'info')
    
    start_time = datetime.datetime.now()
    avg_loss = 0
    for epoch in range(epochs):
      optimizer.zero_grad()
      batch = next(iter(train_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      avg_loss += loss.item()
      loss.backward()
      optimizer.step()
      if(epoch==0):
        custom_print.cprint(f'\t 0% | Epoch {epoch} | Loss: {avg_loss:.4f}', 'wait')
        avg_loss = 0
      elif(100 * epoch / epochs % 5 == 0):
        avg_loss /= (.05*epochs)
        custom_print.cprint(f'\t {100 * epoch / epochs}% | Epoch {epoch} | Loss: {avg_loss:.4f}', 'wait')
        avg_loss = 0
    avg_loss /= (.05*epochs)
    custom_print.cprint(f'\t 100% | Epoch {epoch} | Loss: {avg_loss:.4f}', 'wait')
    custom_print.cprint(f'Final loss: {loss.item():.4f}', 'test')
    
    training_time = datetime.timedelta(seconds=(datetime.datetime.now()-start_time).total_seconds())
    custom_print.cprint(f"Training finished. Total training time (H:mm:ss): {training_time}", 'success')

    # validation
    avg_val_loss = 0
    custom_print.cprint(f"Beginning validation on {test_epochs*batch_size} example sentences (approx. {round(test_epochs/len(test_dataloader), 2)}% of available)...", "info")
    for epoch in range(test_epochs):
      batch = next(iter(test_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      outputs = model(**batch) # unpack the batch dictionary directly into the model
      loss = outputs.loss # the loss is returned when 'labels' are provided in the input
      avg_val_loss += loss.item()
    avg_val_loss /= test_epochs
    custom_print.cprint(f"Validation complete. Avg validation loss: {avg_val_loss}", 'success')

    if(hyperparams.get["save_model_weights"]):
      path = hyperparams.get["save_weights_path"]
      while(os.path.exists(path)): # if the path is already taken
        path += " dupe"
      torch.save(model.state_dict(), f'./saved_models/{path}.pt')
      custom_print.cprint(f"Model weights saved to {path}", "save")
    
    f = open("outputs_summary.txt", "a")
    f.write(f"{date.today()}: {epochs}x{batch_size}, traintime {training_time} --> VLoss {avg_val_loss}")
    dnn_info = "" if config.attention_type == "actual" else f", DNN Layers: {config.DNN_layers}"
    f.write(f"\t attn: {config.attention_type}{dnn_info}, Transfer Learning: {config.load_model_weights} \n")

    f.close()

    return 
    
if __name__ == '__main__': # This code won't run if this file is imported.
  train()