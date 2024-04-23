<<<<<<< HEAD
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
import random
import torch
from torch.utils.data import DataLoader, Dataset

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# we will need to customize this

from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import torch
=======
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader, Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
>>>>>>> 08fdb1ab0d532261b9fbf244fcccc8f549177662

def get_data_loader(sentences, batch_size=32, max_length=32):

    """
    Tokenizes sentences and returns a DataLoader for them.
    :param sentences: List of text sentences.
    :param batch_size: Size of each batch.
    :param max_length: Maximum length of the tokenized output.
    :return: DataLoader with tokenized and appropriately masked batches.
    """
<<<<<<< HEAD
    print("starting tokenizer")
=======

>>>>>>> 08fdb1ab0d532261b9fbf244fcccc8f549177662
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    print("finishd tokenizer")

    
    class SimpleDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

<<<<<<< HEAD
    print('creating dataset')
=======

>>>>>>> 08fdb1ab0d532261b9fbf244fcccc8f549177662
    dataset = SimpleDataset(encoded_inputs)
    print('creating datacollator')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
<<<<<<< HEAD

    print('creating dataloader')
=======
>>>>>>> 08fdb1ab0d532261b9fbf244fcccc8f549177662
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    print('returining')
    return dataloader
    
