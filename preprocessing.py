from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import cuda
import torch.nn as nn
import torch
import numpy as np
import evaluate


device = "cuda" if cuda.is_available() else "cpu"

# import the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# load the dataset
from datasets import load_dataset
dataset = load_dataset("yelp_review_full", split="train")
# joined_data = " ".join([item for inner_list in dataset['set'] for item in inner_list])

#print("jined data", joined_data)



configuration = BertConfig()
model = BertModel(configuration).to(device)
# print(model)


args = TrainingArguments(
    remove_unused_columns=False,
    output_dir=f"temp",
    evaluation_strategy="epoch"
)

dataloader = DataCollatorForLanguageModeling(tokenizer, args, return_tensors="pt")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metricOutput = metric.compute(predictions=predictions, references=labels)
    print("metric", metricOutput)
    return metricOutput


tokenized_datasets = dataset.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))



 
trainer = Trainer(
    model,
    data_collator=dataloader,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)
trainer.train()