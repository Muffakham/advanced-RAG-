#!/bin/bash

# Simple RAG Project Setup Script
# Creates virtual environment, installs dependencies, and sets up OpenAI API key

set -e  # Exit on any error

echo "🚀 Setting up Advanced RAG Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
read -p "Enter your env name (default: venv): " ENV_NAME
ENV_NAME=${ENV_NAME:-venv}

if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Virtual environment '$ENV_NAME' already exists. Removing old one..."
    rm -rf "$ENV_NAME"
fi

python3 -m venv "$ENV_NAME"
echo "✅ Virtual environment '$ENV_NAME' created"

# Activate virtual environment and install dependencies
echo "📥 Installing dependencies..."
source "$ENV_NAME/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Get OpenAI API key and save to .env file
echo ""
echo "🔑 OpenAI API Key Setup"
echo "Get your API key from: https://platform.openai.com/api-keys"
read -p "Enter your OpenAI API key: " OPENAI_KEY

# Create .env file to save the API key
echo "Saving API key to .env file..."
cat > .env << EOF
OPENAI_API_KEY=$OPENAI_KEY
EOF


echo "✅ OpenAI API key saved to .env file and set as environment variable"

# Activate the virtual environment
echo ""
echo "🎉 Setup complete! Activating virtual environment..."
source "$ENV_NAME/bin/activate"
echo "✅ Virtual environment '$ENV_NAME' is now active!"
echo ""
echo "You can now start working with your RAG project!"
