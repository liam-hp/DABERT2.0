get = {

  # training
  "epochs": 5000,
  "test_epochs": 100,
  "batch_size": 32,
  "max_sentence_len": 128,
  "learning_rate": 2e-5,
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