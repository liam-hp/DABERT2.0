import torch
from torch import nn
from transformers import BertForMaskedLM
from typing import Optional, Tuple
import math
import hyperparams

class CustomBertModel(BertForMaskedLM):
    def __init__(self, config):
        super(CustomBertModel, self).__init__(config)
        self.replace_attention_layers(config)
        

    def replace_attention_layers(self, config):
        if hasattr(self, 'bert'): # BertModel is typically the base model for BertForMaskedLM
            encoder_layers = self.bert.encoder.layer
        else:
            raise ValueError("BertForMaskedLM does not contain a 'bert' attribute with an encoder")

        print("config", config)
        for layer in encoder_layers:
            if config.attention_type == "custom":
                layer.attention.self = CustomBertSelfAttention(config)
            elif config.attention_type == "custom_with_values":
                layer.attention.self = CustomBertSelfAttentionWithValues(config)
            elif config.attention_type == "actual":
                layer.attention.self = ActualBertSelfAttention(config)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels  
        )


# use keys after neural net
# this replaces the matrix multiplication in the original implementation
class CustomBertSelfAttentionWithValues(nn.Module):
    def __init__(self, config):
        super(CustomBertSelfAttentionWithValues, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        # projection matracies into each dimension
        # [hidden_size] -> [num_attention_heads x head_size]
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.linearLayers = nn.ModuleList([
        #     nn.Linear(self.attention_head_size * 2, self.attention_head_size), 
        #     # DNN_layers - 1 because we already have one layer
        #     # * is the splat operator and makes a nn.ModuleList of a singular list
        #     ])

        self.linearLayer = nn.Sequential(nn.Linear(self.attention_head_size * 2, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 8192),
                                         nn.ReLU(),
                                         nn.Linear(8192, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, self.attention_head_size))
        

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
        output = torch.cat((key_layer, query_layer), dim=3) 

        # [batch_size x num_heads x sen_length x head_size]
        output = self.linearLayer(output)

        # then softmax and multiply by value layer:

        attention_probs = nn.functional.softmax(output, dim=-1)

        # print("attention shape", attention_probs.shape)
        # print("value shape", value_layer.shape)
        context_layer = torch.matmul(attention_probs, value_layer.transpose(-1, -2))

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

# neural net for everything
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

        layer_factor = hyperparams.get['layer_factor']

        self.linearLayers = nn.ModuleList([
            nn.Linear(self.attention_head_size * 3, layer_factor * self.attention_head_size), 
            # DNN_layers - 1 because we already have one layer
            # * is the splat operator and makes a nn.ModuleList of a singular list
            *[nn.Linear(layer_factor * self.attention_head_size, layer_factor * self.attention_head_size) for _ in range(0, config.DNN_layers - 2)],
            nn.Linear(layer_factor * self.attention_head_size, self.attention_head_size)
            ])

        

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
        output = torch.cat((key_layer, query_layer, value_layer), dim=3) 

        # [batch_size x num_heads x sen_length x head_size]
        for linearLayer in self.linearLayers:
            output = linearLayer(output)

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