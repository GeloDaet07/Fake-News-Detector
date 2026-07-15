# AI Fake News Detector

An AI-powered web application that classifies news articles or text excerpts as "Real News" or "Fake News". 
This project leverages a **DeBERTa** transformer model for text feature extraction and a pre-trained **LightGBM** model for classification.

## Project Architecture
- **Frontend:** Vanilla JS, HTML, and Tailwind CSS. Served via a **Vite** development server for instantaneous Hot Module Replacement (HMR).
- **Backend API:** Python **Flask**. Exposes a `/predict` endpoint that processes the text, extracts embeddings, and returns the LightGBM prediction.

## Downloading the Models (Git LFS)

This project uses Git LFS (Large File Storage) to store the machine learning models. **You must install Git LFS before cloning the repository**, otherwise you will only download small text pointers instead of the actual models, and the backend will fail to load the models.

### Installation

**Mac/Linux:**
```bash
# macOS (using Homebrew)
brew install git-lfs

# Ubuntu/Debian Linux
sudo apt-get install git-lfs
```

**Windows:**
- Download the installer from the [official Git LFS website](https://git-lfs.com/) and run it.
- Alternatively, if you use Winget, run: `winget install -e --id GitHub.GitLFS`

### Fetching the Models

Once Git LFS is installed on your system, initialize it and clone the repository:
```bash
git lfs install
git clone <repository-url>
```
*(Note: If you already cloned the repository without installing Git LFS first, simply install Git LFS and then run `git lfs pull` inside the project folder to download the actual models.)*

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
