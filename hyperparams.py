hyperparams = {

  # training
  "early_stopping": False,
  "epochs": 10,

  # attention specific
  "attention_type": "DNN",
  "DNN_layers": 1,
  "num_encoding_heads": 12,
  "include_key_matrix": NotImplemented,

  # masking
  "predictions_per_mask": 20,
  "masking_rate": .15,
  "context-size": NotImplemented,
  
  # misc
  "disable_bias": NotImplemented,
  "transfer_learning": NotImplemented,
  "next_sentence_prediction": NotImplemented,
  
}