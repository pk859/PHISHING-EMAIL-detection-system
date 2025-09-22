import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ CORRECTED PATHS: Removed "PROJECT EXHIBHITION\" from the start.
MODEL_PATHS = {
    "distilbert": r"Models\distilbert\model\checkpoint-938",
    "bert_base": r"Models\bert_base\model\checkpoint-313",
    "roberta_base": r"Models\roberta_base\model\checkpoint-313",
    "minilm": r"Models\minilm\model\checkpoint-938"
}

# Cache loaded models & tokenizers so they don’t reload every time
_loaded_models = {}

def load_model(model_key):
    """
    Load model and tokenizer for a given model key (distilbert, bert_base, etc.)
    """
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    if model_key not in MODEL_PATHS:
        raise ValueError(f"❌ Unknown model key: {model_key}")

    model_dir = MODEL_PATHS[model_key]
    if not os.path.exists(model_dir):
        # This will now check the correct relative path
        raise FileNotFoundError(f"❌ Model folder not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # ✅ FIX: These lines have been correctly indented to be inside the function.
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map="cpu")

    _loaded_models[model_key] = (model, tokenizer)
    return model, tokenizer