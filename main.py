import random
import torch
from torch import cuda
import torch.optim as optim
import torch.nn as nn
# import early_stopping
import batching
from torch.utils.data import DataLoader
# import format_text
import architecture
from transformers import BertConfig

import preprocessing

device = "cuda" if cuda.is_available() else "cpu"

preprocessing
architecture

def run_bert(attention_type):
    sentences, number_dict, word_dict, token_list, vocab_size = preprocessing.get_training_material()
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = architecture.CustomBertModel(config).to(device) # Single Linear Layer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=8e-7)
    databatches = batching.get_data_loader(sentences)
        # early_stopping = early_stopping.EarlyStopping(tolerance=20, min_delta=0.01) # early stop when 10 occurances where loss does not decrease by > 0.01

    length_sentences, number_dict = preprocessing.get_training_vars()
    testsentences, testnumber_dict = preprocessing.get_testing_output()

    losses = []

    print(f"Initialization complete. Training on {length_sentences} example sentences.")

    sentences, number_dict, word_dict, token_list, vocab_size = preprocessing.get_training_material()

    for batch_num, batch in enumerate(databatches):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the appropriate device
        # outputs = model(**batch, labels=batch['input_ids'].flatten())
        outputs = model(**batch)  # Unpack the batch dictionary directly into the model

        # Compute the loss using the output from the model and the labels
        loss = outputs.loss  # The loss is typically returned when 'labels' are provided in the input
        batchloss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Batch Number: {batch_num} | Loss: {batchloss:.4f}')
        losses.append(batchloss.item())

    averageLast5 = losses[-5:]
    average = sum(averageLast5) / 5
    print("average last 5 losses:", average)
    return average

def main():
    run_bert("us")
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()