import os
import pickle
import re
import traceback
import warnings
import random
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

#You may comment these four lines of code out if you are not using MacOS or if you are not facing issues with forking in multiprocessing
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

app = Flask(__name__)

CODE_DIR = Path(__file__).resolve().parent
MODEL_PATH = CODE_DIR / "models" / "classifiers" / "fnc_bert_lgbm.pkl"
TRANSFORMER_NAME = CODE_DIR / "models" / "transformers" / "DEBERTA-FNC-v2"
TFIDF_SVM_MODEL = CODE_DIR / "models" / "classifiers" / "best_svm.pkl"
TFIDF_SVM_VEC = CODE_DIR / "models" / "vectorizers" / "svm_tfidf_vectorizer.pkl"
TFIDF_LR_MODEL = CODE_DIR / "models" / "classifiers" / "best_lr.pkl"
TFIDF_LR_VEC = CODE_DIR / "models" / "vectorizers" / "lr_tfidf_vectorizer.pkl"
TFIDF_LGBM_MODEL = CODE_DIR / "models" / "classifiers" / "best_tfidf_lgbm.pkl"
TFIDF_LGBM_VEC = CODE_DIR / "models" / "vectorizers" / "lgbm_tfidf_vectorizer.pkl"

BERT_TRANSFORMER_NAME = CODE_DIR / "models" / "transformers" / "FNC-BERT-Fine-tuned"
BERT_LGBM_MODEL_PATH = CODE_DIR / "models" / "classifiers" / "fnc_bert_lgbm_model.pkl"

ROBERTA_TRANSFORMER_NAME = CODE_DIR / "models" / "transformers" / "FNC-RoBERTa-Fine-tuned"
ROBERTA_LGBM_MODEL_PATH = CODE_DIR / "models" / "classifiers" / "fnc_roberta_lgbm_model.pkl"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] Using device: {device}")

model = None
tokenizer = None
transformer_model = None
model_loaded = False
transformer_loaded = False

svm_model = None
svm_vec = None
svm_loaded = False

lr_model = None
lr_vec = None
lr_loaded = False

lgbm_model_tfidf = None
lgbm_vec_tfidf = None
lgbm_tfidf_loaded = False

deberta_classifier = None
deberta_classifier_loaded = False

bert_tokenizer = None
bert_transformer_model = None
bert_lgbm = None
bert_loaded = False

roberta_tokenizer = None
roberta_transformer_model = None
roberta_lgbm = None
roberta_loaded = False

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

if os.path.exists(TFIDF_SVM_MODEL) and os.path.exists(TFIDF_SVM_VEC):
    try:
        with open(TFIDF_SVM_MODEL, 'rb') as f:
            svm_model = pickle.load(f)
        with open(TFIDF_SVM_VEC, 'rb') as f:
            svm_vec = pickle.load(f)
        svm_loaded = True
        print(f"[*] Successfully loaded TF-IDF SVM model.")
    except Exception as e:
        print(f"[!] Error loading TF-IDF SVM model: {e}")
else:
    print(f"[!] Warning: TF-IDF SVM model files not found.")

if os.path.exists(TFIDF_LR_MODEL) and os.path.exists(TFIDF_LR_VEC):
    try:
        with open(TFIDF_LR_MODEL, 'rb') as f:
            lr_model = pickle.load(f)
        with open(TFIDF_LR_VEC, 'rb') as f:
            lr_vec = pickle.load(f)
        lr_loaded = True
        print(f"[*] Successfully loaded TF-IDF LR model.")
    except Exception as e:
        print(f"[!] Error loading TF-IDF LR model: {e}")
else:
    print(f"[!] Warning: TF-IDF LR model files not found.")

if os.path.exists(TFIDF_LGBM_MODEL) and os.path.exists(TFIDF_LGBM_VEC):
    try:
        with open(TFIDF_LGBM_MODEL, 'rb') as f:
            lgbm_model_tfidf = pickle.load(f)
        with open(TFIDF_LGBM_VEC, 'rb') as f:
            lgbm_vec_tfidf = pickle.load(f)
        lgbm_tfidf_loaded = True
        print(f"[*] Successfully loaded TF-IDF LGBM model.")
    except Exception as e:
        print(f"[!] Error loading TF-IDF LGBM model: {e}")
else:
    print(f"[!] Warning: TF-IDF LGBM model files not found.")

print(f"[*] Loading DeBERTa Classifier ({TRANSFORMER_NAME}). This may take a moment...")
try:
    if os.path.exists(TRANSFORMER_NAME):
        deberta_classifier = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_NAME), local_files_only=True)
        deberta_classifier.to(device)
        deberta_classifier.eval()
        deberta_classifier_loaded = True
        print("[*] Successfully loaded DeBERTa Classifier.")
    else:
        print(f"[!] Warning: Transformer folder not found for classifier: {TRANSFORMER_NAME}")
except Exception as e:
    print(f"[!] Error loading DeBERTa Classifier: {e}")

print(f"[*] Loading BERT model ({BERT_TRANSFORMER_NAME}). This may take a moment...")
try:
    if os.path.exists(BERT_TRANSFORMER_NAME) and os.path.exists(BERT_LGBM_MODEL_PATH):
        bert_tokenizer = AutoTokenizer.from_pretrained(str(BERT_TRANSFORMER_NAME), local_files_only=True)
        bert_transformer_model = AutoModel.from_pretrained(str(BERT_TRANSFORMER_NAME), local_files_only=True)
        bert_transformer_model.to(device)
        bert_transformer_model.eval()
        with open(BERT_LGBM_MODEL_PATH, 'rb') as f:
            bert_lgbm = pickle.load(f)
        bert_loaded = True
        print("[*] Successfully loaded BERT + LightGBM models.")
    else:
        print("[!] Warning: BERT models not found.")
except Exception as e:
    print(f"[!] Error loading BERT models: {e}")

print(f"[*] Loading RoBERTa model ({ROBERTA_TRANSFORMER_NAME}). This may take a moment...")
try:
    if os.path.exists(ROBERTA_TRANSFORMER_NAME) and os.path.exists(ROBERTA_LGBM_MODEL_PATH):
        roberta_tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_TRANSFORMER_NAME), local_files_only=True)
        roberta_transformer_model = AutoModel.from_pretrained(str(ROBERTA_TRANSFORMER_NAME), local_files_only=True)
        roberta_transformer_model.to(device)
        roberta_transformer_model.eval()
        with open(ROBERTA_LGBM_MODEL_PATH, 'rb') as f:
            roberta_lgbm = pickle.load(f)
        roberta_loaded = True
        print("[*] Successfully loaded RoBERTa + LightGBM models.")
    else:
        print("[!] Warning: RoBERTa models not found.")
except Exception as e:
    print(f"[!] Error loading RoBERTa models: {e}")

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
    start_time = time.time()
    try:
        data = request.json or {}
        raw_text = data.get('text', '')
        selected_model = data.get('model', 'deberta-lgbm')

        if not raw_text:
            return jsonify({'error': 'No text provided'}), 400

        cleaned_text = clean_text(raw_text)

        if selected_model == 'deberta-lgbm':
            if not model_loaded or not transformer_loaded:
                is_fake = random.choice([True, False])
                confidence = random.uniform(0.65, 0.99) * 100
                label = "Fake News" if is_fake else "Real News"
                prob_fake = confidence if is_fake else 100.0 - confidence
                prob_real = 100.0 - confidence if is_fake else confidence
                inference_time_sec = time.time() - start_time
                return jsonify({
                    'prediction': label,
                    'confidence': confidence,
                    'prob_real': prob_real,
                    'prob_fake': prob_fake,
                    'inference_time_sec': inference_time_sec,
                    'raw_pred': "MOCK",
                    'mock': True
                })

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

        elif selected_model == 'tfidf-svm':
            if not svm_loaded:
                return jsonify({'error': 'TF-IDF SVM model not loaded on server.'}), 500
            
            vec_text = svm_vec.transform([cleaned_text])
            pred = svm_model.predict(vec_text)[0]
            
            if hasattr(svm_model, "predict_proba"):
                probs = svm_model.predict_proba(vec_text)[0]
                confidence = max(probs) * 100
            else:
                confidence = 100.0

        elif selected_model == 'tfidf-lr':
            if not lr_loaded:
                return jsonify({'error': 'TF-IDF LR model not loaded on server.'}), 500
            
            vec_text = lr_vec.transform([cleaned_text])
            pred = lr_model.predict(vec_text)[0]
            
            if hasattr(lr_model, "predict_proba"):
                probs = lr_model.predict_proba(vec_text)[0]
                confidence = max(probs) * 100
            else:
                confidence = 100.0
                
        elif selected_model == 'tfidf-lgbm':
            if not lgbm_tfidf_loaded:
                return jsonify({'error': 'TF-IDF LGBM model not loaded on server.'}), 500
            
            vec_text = lgbm_vec_tfidf.transform([cleaned_text])
            pred = lgbm_model_tfidf.predict(vec_text)[0]
            
            if hasattr(lgbm_model_tfidf, "predict_proba"):
                probs = lgbm_model_tfidf.predict_proba(vec_text)[0]
                confidence = max(probs) * 100
            else:
                confidence = 100.0

        elif selected_model == 'deberta-classifier':
            if not deberta_classifier_loaded:
                return jsonify({'error': 'DeBERTa Classifier not loaded on server.'}), 500
            
            encodings = tokenizer(
                [cleaned_text],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = deberta_classifier(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred = str(pred_idx)
                confidence = float(max(probs) * 100)
                
        elif selected_model == 'bert-lgbm':
            if not bert_loaded:
                return jsonify({'error': 'BERT models not loaded on server.'}), 500
            
            embeddings = extract_embeddings(
                texts=[cleaned_text],
                tokenizer=bert_tokenizer,
                model=bert_transformer_model,
                device=device,
                pooling='cls',
                hidden_layers='last',
                batch_size=1
            )
            
            pred = bert_lgbm.predict(embeddings)[0]
            if hasattr(bert_lgbm, "predict_proba"):
                probs = bert_lgbm.predict_proba(embeddings)[0]
                confidence = max(probs) * 100
            else:
                confidence = 100.0

        elif selected_model == 'roberta-lgbm':
            if not roberta_loaded:
                return jsonify({'error': 'RoBERTa models not loaded on server.'}), 500
            
            embeddings = extract_embeddings(
                texts=[cleaned_text],
                tokenizer=roberta_tokenizer,
                model=roberta_transformer_model,
                device=device,
                pooling='cls',
                hidden_layers='last',
                batch_size=1
            )
            
            pred = roberta_lgbm.predict(embeddings)[0]
            if hasattr(roberta_lgbm, "predict_proba"):
                probs = roberta_lgbm.predict_proba(embeddings)[0]
                confidence = max(probs) * 100
            else:
                confidence = 100.0
                
        else:
            return jsonify({'error': 'Invalid model selected.'}), 400

        if str(pred) in ['1', 'Fake', 'False', 'false']:
            prediction_label = "Fake News"
            prob_fake = float(confidence)
            prob_real = 100.0 - prob_fake
        else:
            prediction_label = "Real News"
            prob_real = float(confidence)
            prob_fake = 100.0 - prob_real
            
        inference_time_sec = time.time() - start_time

        return jsonify({
            'prediction': prediction_label,
            'confidence': float(confidence),
            'prob_real': prob_real,
            'prob_fake': prob_fake,
            'inference_time_sec': inference_time_sec,
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
