get = {

  # training
  "epochs": 10000,
  "test_epochs": 100,
  "batch_size": 32,
  "max_sentence_len": 32,
  "learning_rate": NotImplemented,
  "adaptive_lr": NotImplemented,
  "early_stopping": NotImplemented,

  # attention specific
  "attention_type": NotImplemented,
  "DNN_layers": NotImplemented,
  "num_encoding_heads": NotImplemented,
  "include_key_matrix": NotImplemented,

  # masking
  "predictions_per_mask": NotImplemented,
  "masking_rate": NotImplemented,
  "context-size": NotImplemented,
  
  # misc
  "disable_bias": NotImplemented,
  "transfer_learning": NotImplemented,
  "next_sentence_prediction": NotImplemented,
  
}