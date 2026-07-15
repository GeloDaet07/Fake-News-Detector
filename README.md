# AI Fake News Detector

An AI-powered web application that classifies news articles or text excerpts as "Real News" or "Fake News". 
This project leverages a **DeBERTa** transformer model for text feature extraction and a pre-trained **LightGBM** model for classification.

## Project Architecture
- **Frontend:** Vanilla JS, HTML, and Tailwind CSS. Served via a **Vite** development server for instantaneous Hot Module Replacement (HMR).
- **Backend API:** Python **Flask**. Exposes a `/predict` endpoint that processes the text, extracts embeddings, and returns the LightGBM prediction.

## How to Run the Application

Because this project uses a split architecture (Vite for the frontend, Flask for the backend API), you need to run both servers simultaneously.

### Step 1: Start the Backend API (Terminal 1)
Open your terminal, navigate to the project root directory, and activate the virtual environment. Then start the Flask server.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the Flask API
python backend/app.py
```
*The backend API will start on `http://localhost:5001`.*

### Step 2: Start the Frontend Vite Server (Terminal 2)
Open a **new** terminal window, navigate to the project root directory, and start the Vite development server.

```bash
# Start the Vite server
npm run dev
```
*Vite will start the frontend on `http://localhost:5173`. Vite is configured to automatically proxy any `/predict` API requests to your Python backend.*

### Step 3: Use the App
Open your web browser and navigate to **[http://localhost:5173](http://localhost:5173)**. Paste a news excerpt and click "Analyze Authenticity"!

## Testing
To verify that the model logic works correctly:
```bash
source .venv/bin/activate
cd backend
python test_app.py
```
