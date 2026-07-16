#!/bin/bash

# Setup script for AI Fake News Detector
# This script installs all necessary frontend and backend dependencies.

echo "================================================="
echo "   Setting up the AI Fake News Detector...       "
echo "================================================="

# 1. Check Node.js
if ! command -v npm &> /dev/null
then
    echo "[!] npm could not be found. Please install Node.js first."
    exit 1
fi

# 2. Check Python
if ! command -v python3 &> /dev/null
then
    echo "[!] python3 could not be found. Please install Python 3 first."
    exit 1
fi

echo "[*] Installing frontend dependencies (Node.js/npm)..."
npm install
if [ $? -ne 0 ]; then
    echo "[!] Failed to install frontend dependencies."
    exit 1
fi

echo "[*] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "[*] Created .venv directory."
else
    echo "[*] .venv already exists."
fi

echo "[*] Activating virtual environment and installing backend dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[!] Failed to install Python backend dependencies."
    exit 1
fi

echo "================================================="
echo "   Setup Complete!                               "
echo "================================================="
echo "You can now run the app with:"
echo "Frontend: npm run dev"
echo "Backend:  source .venv/bin/activate && python backend/app.py"
echo "================================================="
