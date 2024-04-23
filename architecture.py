import torch
from torch import nn
from transformers import BertForMaskedLM, BertModel
from typing import List, Optional, Tuple, Union
import math

class CustomBertModel(BertForMaskedLM):
    def __init__(self, config):
        super(CustomBertModel, self).__init__(config)
        self.replace_attention_layers(config)
        

    def replace_attention_layers(self, config):
        if hasattr(self, 'bert'): # BertModel is typically the base model for BertForMaskedLM
            encoder_layers = self.bert.encoder.layer
        else:
            raise ValueError("BertForMaskedLM does not contain a 'bert' attribute with an encoder")

        for layer in encoder_layers:
            layer.attention.self = CustomBertSelfAttention(config)
            #layer.attention.self = ActualBertSelfAttention(config)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels  
        )

class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(CustomBertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # projection matracies into each dimension
        # [hidden_size] -> [num_attention_heads x head_size]
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.linearLayer = nn.Linear(self.attention_head_size * 3, self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, 
                hidden_states, 
                attention_mask=None, 
                head_mask=None, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None, past_key_value=None, output_attentions=None):

        mixed_query_layer = self.query(hidden_states)

        # projects each kvq into a different head 
        # [batch_size x sen_length x hidden_size] 
        # -> 
        # [batch_size x num_heads x sen_length x head_size]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # combine on the last dimension
        combined = torch.cat((key_layer, query_layer, value_layer), dim=3) 

        # [batch_size x num_heads x sen_length x head_size]
        output = self.linearLayer(combined)

        # Begin transofrmation to original shape

        # 1) Reorder the tensor to [batch_size x sen_length x num_heads x head_size]
        context_layer = output.permute(0, 2, 1, 3).contiguous()

        # 2) Computer output format: [batch_size x sen_length x hidden_size]
        #    context_layer.size()[:-2] gets [batch_size x sen_length] 
        #    self.all_head_size is the hidden_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # 3) Reshape context to: [batch_size x sen_length x hidden_size]
        context_layer = context_layer.view(new_context_layer_shape)

        # Special output format specified by origional Bert implementation
        outputs = (context_layer,)
        # Reshape context to: [batch_size x sen_length x hidden_size]
        return outputs



# the actual implementation of the attention model
class ActualBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(ActualBertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # projection matracies into each dimension
        # [hidden_size] -> [num_attention_heads x head_size]
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)


        # projects each kvq into a different head 
        # [batch_size x sen_length x hidden_size] 
        # -> 
        # [batch_size x num_heads x sen_length x head_size]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)


        # Take the dot product between q and k to get the raw attention scores.
        # [batch_size x num_heads x sen_length x sen_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Multiply by value layer
        # [batch_size x num_heads x sen_length x head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # Begin transofrmation to original shape

        # 1) Reorder the tensor to [batch_size x sen_length x num_heads x head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 2) Computer output format: [batch_size x sen_length x hidden_size]
        #    context_layer.size()[:-2] gets [batch_size x sen_length] 
        #    self.all_head_size is the hidden_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        # 3) Reshape context to: [batch_size x sen_length x hidden_size]
        context_layer = context_layer.view(new_context_layer_shape)


        # Special output format specified by origional Bert implementation
        outputs = (context_layer,)

        return outputs