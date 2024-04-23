import torch
from torch import nn
from transformers import BertForMaskedLM, BertModel

class CustomBertModel(BertForMaskedLM):
    def __init__(self, config):
        super(CustomBertModel, self).__init__(config)
        # Ensure encoder layers are properly initialized before modifying them
        self.replace_attention_layers(config)
        

    def replace_attention_layers(self, config):
        # Access the base model's encoder to replace its attention layers
        # BertModel is typically the base model for BertForMaskedLM
        if hasattr(self, 'bert'):
            encoder_layers = self.bert.encoder.layer
        else:
            raise ValueError("BertForMaskedLM does not contain a 'bert' attribute with an encoder")

        for layer in encoder_layers:
            layer.attention.self = CustomAttention(config)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Forward call to the superclass method, passing only expected args
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels  # Ensure you handle labels correctly if using them for loss calculation
        )

class CustomAttention(nn.Module):
    def __init__(self, config):
        super(CustomAttention, self).__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.combine = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Custom attention mechanism implementation
        query_layer = self.query(hidden_states)
        value_layer = self.value(hidden_states)
        combined = torch.cat((query_layer, value_layer), dim=-1)  # Ensure correct dimension
        output = self.combine(combined)
        return output
