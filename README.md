# DABERT2.0
Replacing BERTs attention layer with a DNN in an attempt to encode Query Value relationships in a more complex way, hopefully capturing more context.

# Reproducibility

Due to the large number of dependencies we suggest using a virtualenv: 
```shell
python3 -m venv venv
source venv/bin/activate
```

We use python version `3.9`. 
1) Install the dependencies:
```shell
pip install -r requirements.txt
```

2) Adjust the hyperparams as desired
Our reported data comes from the folllowing configuration in the `hyperparams.py` file
```python
get = {

  # saving and loading model weights as .pt, so excluded from git uploads in .gitignore
  "save_model_weights": False,
  "save_weights_path": "",

  "load_model_weights": False,
  "load_weights_path": "save-dynamic-2e-5-actual",

  # training hyperparameters
  "epochs": 50000
  "test_epochs": 100,
  "batch_size": 32,
  "max_sentence_len": 32,
  "learning_rate": 2e-5,

  # use the attention in the origional Attention Is All You Need paper
  "attention_type": "actual", 
  # use the custom attention mechanism which concatenates the K, V, Q matricies 
  # "attention_type": "custom", 
  # use the custom attention mechanism which concatenates the K, Q matricies and multiplies by V. 
  # "attention_type": "custom_with_values", 

  # custom number of DNN layers
  "DNN_layers": 10,
  "layer_factor": 2,
  
  "architectures": [
    "BertForMaskedLM"
  ],
  # all default configuration from the Huggingfase BertConfig
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": None,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.36.2",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 30522
}
```

3) Run the `main.py` file:
```shell
python main.py
```
4) Monitor your terminal for periodic initialization, training, and validation updates from the model.
