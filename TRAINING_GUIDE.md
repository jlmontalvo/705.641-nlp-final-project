# Complete Training Guide

This guide will walk you through training the models from scratch.

## Prerequisites

1. **Data**: You need the preprocessed data files:
   - `train_data.parquet`
   - `val_data.parquet`
   - `test_data.parquet`

   These are created by running the data preprocessing notebook (`nlp_final_proj_data_prep.ipynb`).

2. **Hardware**: 
   - GPU recommended (CUDA-capable)
   - At least 16GB RAM
   - 50GB+ free disk space

3. **Dependencies**: All packages from `requirements.txt`

## Step-by-Step Training

### Step 1: Prepare Your Data

If you haven't already, run the data preprocessing notebook to create the parquet files:

1. Open `nlp_final_proj_data_prep.ipynb`
2. Run all cells to:
   - Load and clean the data
   - Generate StyleDistance embeddings
   - Split into train/val/test sets
   - Save as parquet files

**Output**: You should have:
- `train_data.parquet` (~400K samples)
- `val_data.parquet` (~50K samples)
- `test_data.parquet` (~50K samples)

### Step 2: Organize Your Data

Create a data directory and place your parquet files:

```bash
mkdir -p data
# Copy your parquet files to data/
cp /path/to/train_data.parquet data/
cp /path/to/val_data.parquet data/
cp /path/to/test_data.parquet data/
```

### Step 3: Train the Models

#### Option A: Train Both Models (Recommended)

Use the main training script:

```bash
python train_model.py \
    --data-dir ./data \
    --output-dir ./models \
    --bert-epochs 3 \
    --classifier-epochs 10 \
    --batch-size 8 \
    --learning-rate 2e-5
```

This will:
1. Fine-tune BERT (takes ~2-4 hours on GPU)
2. Extract embeddings
3. Train integrated classifier (takes ~1-2 hours on GPU)

#### Option B: Train Models Separately

**Train BERT first:**

```python
from src.training.train_bert import train_bert_model

bert_path = train_bert_model(
    data_dir='./data',
    output_dir='./models',
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5
)
```

**Then train the classifier:**

```python
from src.training.train_classifier import train_integrated_classifier

classifier_path = train_integrated_classifier(
    data_dir='./data',
    bert_model_path=bert_path,
    output_dir='./models',
    num_epochs=10,
    batch_size=32
)
```

### Step 4: Verify Models

After training, you should have:

```
models/
├── bert_checkpoint/
│   ├── final/          # Fine-tuned BERT model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files
│   └── checkpoint-*/    # Epoch checkpoints
└── integrated_classifier_best.pth  # Trained classifier
```

### Step 5: Use Your Trained Models

Set environment variables:

```bash
export BERT_MODEL_PATH="./models/bert_checkpoint/final"
export CLASSIFIER_MODEL_PATH="./models/integrated_classifier_best.pth"
```

Or add to `.env`:

```bash
BERT_MODEL_PATH=./models/bert_checkpoint/final
CLASSIFIER_MODEL_PATH=./models/integrated_classifier_best.pth
```

Then run the application:

```bash
./run.sh
```

## Training Parameters

### BERT Fine-tuning

- **Epochs**: 3 (recommended)
- **Batch size**: 8 (adjust based on GPU memory)
- **Learning rate**: 2e-5 (standard for BERT)
- **Max length**: 512 tokens

### Integrated Classifier

- **Epochs**: 10 (usually converges by epoch 6-8)
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Architecture**: 1536 → 512 → 256 → 2

## Expected Results

With proper training, you should achieve:

- **BERT fine-tuning**: ~90-95% validation accuracy
- **Integrated classifier**: ~96-97% validation accuracy
- **Test accuracy**: ~96.7%

## Troubleshooting

### Out of Memory Errors

- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use gradient accumulation
- Process embeddings in smaller batches

### Slow Training

- Use GPU if available
- Reduce number of training samples for testing
- Use mixed precision training (fp16)

### Poor Results

- Ensure data preprocessing was done correctly
- Check that StyleDistance embeddings were generated
- Verify data quality and balance
- Try training for more epochs
- Adjust learning rate

### Model Not Saving

- Check disk space
- Verify write permissions
- Ensure output directory exists

## Training Time Estimates

On a modern GPU (e.g., NVIDIA RTX 3090):

- **BERT fine-tuning**: 2-4 hours (3 epochs, 400K samples)
- **Embedding extraction**: 1-2 hours
- **Classifier training**: 1-2 hours (10 epochs)
- **Total**: ~4-8 hours

On CPU, expect 5-10x longer.

## Monitoring Training

Watch the logs for:
- Training/validation loss (should decrease)
- Accuracy (should increase)
- Best model checkpoints

The script will automatically save the best model based on validation accuracy.

## Next Steps

After training:

1. Evaluate on test set (optional)
2. Set model paths in environment
3. Test the inference pipeline
4. Deploy the application

## Quick Start Example

```bash
# 1. Prepare data (run preprocessing notebook first)
mkdir -p data
# Copy parquet files to data/

# 2. Train models
python train_model.py --data-dir ./data --output-dir ./models

# 3. Set paths
export BERT_MODEL_PATH="./models/bert_checkpoint/final"
export CLASSIFIER_MODEL_PATH="./models/integrated_classifier_best.pth"

# 4. Run application
./run.sh
```

That's it! You now have trained models achieving 96.7% accuracy! 🎉

