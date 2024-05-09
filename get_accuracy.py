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

device = "cuda" if cuda.is_available() else "cpu"
custom_print.cprint(f"Setting device... {device}", 'setting')

def get_acc(model_path):
    
    batch_size = hyperparams.get["batch_size"]
    max_sent_len = hyperparams.get["max_sentence_len"]

    custom_print.cprint("Initializing config, model, and optimizer...", "setting")
    config = PretrainedConfig.from_dict(copyHyperparams)

    model = architecture.CustomBertModel(config).to(device)

    # load in pretrained weights
    
    if(os.path.exists(f"./saved_models/{model_path}.pt")): # if the path exists
      model_state_dict = model.state_dict() # save the initialized state dict of our model
      loaded_sd = torch.load(f'./saved_models/{model_path}.pt') # load in the saved dict of a previous model
      # keep only the loaded_sd keys that match keys in init_state_dict
      filtered_sd = {
        k: v for k, v in loaded_sd.items() if k in model_state_dict and 
                                              v.size() == model_state_dict[k].size()
      } 
      model_state_dict.update(filtered_sd)  # update existing layers in the init dict with the loaded dict
      model.load_state_dict(model_state_dict) # load the modified state_dict; uninitialized weights remain randomly initialized
    else:
      custom_print.cprint("Failed to load model weights", "save")


    acc_dataloader = batching.get_data_loader(sentences, batch_size=batch_size, max_length=max_sent_len)

    # validation
    custom_print.cprint(f"Beginning accuracy calculation)...", "info")
    model.eval()
    total_accuracy = 0
    n_batches = 0

    for _ in range(1000):
      batch = next(iter(acc_dataloader))# batch is a dict of keys: input_ids, token_type_ids, attention_mask, labels
      batch = {k: v.to(device) for k, v in batch.items()} # move batch to the appropriate device
      with torch.no_grad():
        outputs = model(**batch) # unpack the batch dictionary directly into the model
        predictions = outputs.logits  # Assuming that the model returns logits
        labels = batch['labels']
        mask = labels != -100  # Assuming that labels for non-masked tokens are set to -100
        accuracy = compute_accuracy(predictions, labels, mask)
        total_accuracy += accuracy
        n_batches += 1

    average_accuracy = total_accuracy / n_batches
    custom_print.cprint(f'Average MLM Accuracy: {average_accuracy * 100:.2f}%', "success")

    return 

def compute_accuracy(predictions, labels, mask):
    # Only consider the masked positions
    predictions = predictions[mask.bool()].argmax(dim=-1)
    labels = labels[mask.bool()]
    correct_predictions = (predictions == labels).float().sum()
    total_predictions = mask.sum()
    return (correct_predictions / total_predictions).item()

if __name__ == '__main__': # This code won't run if this file is imported.
  get_acc(model_path="save-dynamic+100-2e-5-actualxcustom")