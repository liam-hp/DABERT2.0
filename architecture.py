import torch
from torch import nn
from transformers import BertForMaskedLM

class CustomBertModel(BertForMaskedLM):
    def __init__(self, config):

        super(CustomBertModel, self).__init__(config)
        self.replace_attention_layers(config)
        

    def replace_attention_layers(self, config):

        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers:
            layer.attention.self = CustomAttention(config)


    # def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
    def forward(self):

        # return super().forward(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
        return super().forward()

class CustomAttention(nn.Module):
    def __init__(self, config):

        super(CustomAttention, self).__init__()
        # print("init custom atteition layer", config)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.combine = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, 
                hidden_states, 
                attention_mask=None, 
                head_mask=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None, 
                past_key_value=None, output_attentions=None
                ):
        query_layer = self.query(hidden_states)
        value_layer = self.value(hidden_states)
        combined = torch.cat((query_layer, value_layer), dim=-1)  # ensure correct dimension
        output = self.combine(combined)
        return (output, output)


