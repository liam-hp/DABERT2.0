from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from datasets import load_dataset
import batches
from torch.utils.data import DataLoader
import preprocess
from torch import cuda
import torch.nn as nn
import torch

configuration = BertConfig()


def get_model():
    model = BertModel(configuration)
    return model
# print(model)
