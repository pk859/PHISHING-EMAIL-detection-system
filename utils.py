import torch
import torch.nn.functional as F
# ✅ CORRECTED IMPORT: Changed 'model' to 'models' to match your file name.
from models import load_model

def clean_text(text: str) -> str:
    """
    Simple cleaning for input email text.
    (You can expand this with regex, stopwords removal, etc.)
    """
    return text.strip().lower()

def predict_email(model_key: str, text: str) -> dict:
    """
    Run prediction on given text using chosen model.
    Returns label + probability.
    """
    model, tokenizer = load_model(model_key)
    model.eval()

    inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    return {
        "label": "Phishing" if pred_label == 1 else "Safe",
        "confidence": round(confidence, 4),
        "probabilities": probs.tolist()
    }