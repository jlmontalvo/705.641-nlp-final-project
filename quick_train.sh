#!/bin/bash

# Quick training script - trains both models
# Usage: ./quick_train.sh [data_dir] [output_dir]

set -e

DATA_DIR=${1:-./data}
OUTPUT_DIR=${2:-./models}

echo "="*70
echo "Quick Training Script"
echo "="*70
echo ""
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    echo ""
    echo "Please:"
    echo "1. Run the data preprocessing notebook to create parquet files"
    echo "2. Place train_data.parquet, val_data.parquet, test_data.parquet in $DATA_DIR"
    echo "3. Or specify a different data directory: ./quick_train.sh /path/to/data"
    exit 1
fi

# Check for required files
for file in train_data.parquet val_data.parquet; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Error: Required file '$DATA_DIR/$file' not found!"
        echo "Please run the data preprocessing notebook first."
        exit 1
    fi
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import torch, transformers, datasets, sentence_transformers" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
}

echo ""
echo "Starting training..."
echo "This will take several hours. You can monitor progress in the logs."
echo ""

# Run training
python train_model.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --bert-epochs 3 \
    --classifier-epochs 10 \
    --batch-size 8 \
    --learning-rate 2e-5

echo ""
echo "="*70
echo "Training completed!"
echo "="*70
echo ""
echo "To use your trained models, set:"
echo "  export BERT_MODEL_PATH='$OUTPUT_DIR/bert_checkpoint/final'"
echo "  export CLASSIFIER_MODEL_PATH='$OUTPUT_DIR/integrated_classifier_best.pth'"
echo ""
echo "Or add to .env file:"
echo "  BERT_MODEL_PATH=$OUTPUT_DIR/bert_checkpoint/final"
echo "  CLASSIFIER_MODEL_PATH=$OUTPUT_DIR/integrated_classifier_best.pth"
echo ""

