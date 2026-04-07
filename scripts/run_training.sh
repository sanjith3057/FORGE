#!/bin/bash
set -e

echo "🔥 FORGE — Starting QLoRA Pipeline"
echo "=================================="

# Check GPU capability
echo "[Hardware Check]"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU detected.')"

echo "[Step 1] Initializing Directories"
mkdir -p outputs/adapters outputs/merged logs results

echo "[Step 2] Beginning Fine-tuning"
# Run the training script configured for local RTX 2050 bounds
python -m src.train

echo "[Step 3] Interface Testing"
echo "Launch Streamlit with: streamlit run ui/app.py"
