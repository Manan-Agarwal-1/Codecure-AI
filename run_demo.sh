#!/bin/bash

echo "🚀 Setting up AI Epidemic Forecasting Dashboard..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run data pipeline
echo "🔄 Processing data..."
python data_pipeline.py

# Train models
echo "🤖 Training AI models..."
python models/train_model.py

# Run dashboard
echo "🌟 Launching dashboard..."
echo "Dashboard will be available at http://localhost:8501"
streamlit run dashboard/app.py --server.headless true --server.port 8501