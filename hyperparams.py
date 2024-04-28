get = {

  # training
  "epochs": 50000,
  "test_epochs": 100,
  "batch_size": 32,
  "max_sentence_len": 32,
  "learning_rate": NotImplemented,
  "adaptive_lr": NotImplemented,
  "early_stopping": NotImplemented,

  # attention specific
  # custom or actual
  "attention_type": "custom_with_values",
  "DNN_layers": 1,
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
  
  "architectures": [
    "BertForMaskedLM"
  ],
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