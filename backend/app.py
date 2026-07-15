import os
import pickle
import re
import traceback
import warnings
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

app = Flask(__name__)

CODE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CODE_DIR / "models" / "new_lgbm_model_optuna.pkl"
TRANSFORMER_NAME = CODE_DIR / "models" / "DEBERTA-FNC-v2"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] Using device: {device}")

model = None
tokenizer = None
transformer_model = None
model_loaded = False
transformer_loaded = False

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print(f"[*] Loading Transformer model ({TRANSFORMER_NAME}). This may take a moment...")
try:
    if os.path.exists(TRANSFORMER_NAME):
        tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_NAME), local_files_only=True)
        transformer_model = AutoModel.from_pretrained(str(TRANSFORMER_NAME), local_files_only=True)
        transformer_model.to(device)
        transformer_model.eval()
        transformer_loaded = True
        print("[*] Successfully loaded Transformer model.")
    else:
        print(f"[!] Warning: Transformer folder not found: {TRANSFORMER_NAME}")
except Exception as e:
    print(f"[!] Error loading Transformer model: {e}")

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
        print(f"[*] Successfully loaded LightGBM model: {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Error loading LightGBM model: {e}")
else:
    print(f"[!] Warning: LightGBM model file not found: {MODEL_PATH}")

def extract_embeddings(texts, tokenizer, model, device, pooling='cls', hidden_layers='last', batch_size=16):
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        encodings = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            all_hidden = outputs.hidden_states

            if hidden_layers == 'last':
                layers_to_use = [all_hidden[-1]]
            elif hidden_layers == 'last4':
                layers_to_use = list(all_hidden[-4:])
            else:
                raise ValueError(f"Unsupported hidden_layers value: {hidden_layers}")

            batch_embeds = []
            for layer_hidden in layers_to_use:
                if pooling == 'cls':
                    pooled = layer_hidden[:, 0, :]
                elif pooling == 'mean':
                    mask = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                    sum_hidden = torch.sum(layer_hidden * mask, dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = sum_hidden / counts
                else:
                    raise ValueError(f"Unsupported pooling value: {pooling}")
                batch_embeds.append(pooled)

            if hidden_layers == 'last4':
                batch_combined = torch.cat(batch_embeds, dim=-1)
            else:
                batch_combined = batch_embeds[0]

            all_embeddings.append(batch_combined.cpu().numpy())

    return np.vstack(all_embeddings)

@app.route('/')
def home():
    return jsonify({"message": "Backend API is running. Please use the Vite dev server (port 5173) for the frontend."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        raw_text = data.get('text', '')

        if not raw_text:
            return jsonify({'error': 'No text provided'}), 400

        if not model_loaded or not transformer_loaded:
            is_fake = random.choice([True, False])
            confidence = random.uniform(0.65, 0.99) * 100
            label = "Fake News" if is_fake else "Real News"
            return jsonify({
                'prediction': label,
                'confidence': confidence,
                'raw_pred': "MOCK",
                'mock': True
            })

        cleaned_text = clean_text(raw_text)

        embeddings = extract_embeddings(
            texts=[cleaned_text],
            tokenizer=tokenizer,
            model=transformer_model,
            device=device,
            pooling='cls',
            hidden_layers='last',
            batch_size=1
        )

        pred = model.predict(embeddings)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(embeddings)[0]
            confidence = max(probs) * 100
        else:
            confidence = 100.0

        if str(pred) in ['1', 'Fake', 'False', 'false']:
            prediction_label = "Fake News"
        else:
            prediction_label = "Real News"

        return jsonify({
            'prediction': prediction_label,
            'confidence': float(confidence),
            'raw_pred': str(pred),
            'mock': False
        })

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': 'Error processing text. Check the terminal or notebook output for details.'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'transformer_loaded': transformer_loaded,
        'model_path': str(MODEL_PATH),
        'transformer_path': str(TRANSFORMER_NAME),
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5001'))
    host = os.environ.get('HOST', '127.0.0.1')
    
    print("\n--- Fake News Detector Server Starting ---")
    print(f"Open http://localhost:{port} in your browser")
    
    app.run(host=host, port=port, debug=True, use_reloader=False)
