# AI-Generated Text Detection System

## EN 705.641 NLP Final Project

**Authors:** Jose Montalvo Ferreiro, Muhammad Khan, Joe Mfonfu, Madihah Shaik

[Presentation Slides](https://docs.google.com/presentation/d/1B-RdgZzLafXHr7mEVduqDNgqa0cGlSDPfSTU-bc7OqM/edit?usp=sharing)

---

## 📋 About the Project

This project implements a state-of-the-art system for distinguishing between human-written and AI-generated text using a **hybrid approach** that integrates semantic and stylistic features. The system achieves **>96% accuracy** on the Human Vs. LLM Text Corpus dataset.

### Key Features

- **Hybrid Feature Extraction**: Combines semantic features from fine-tuned BERT and stylistic features from StyleDistance
- **High Accuracy**: Achieves 96.7% accuracy on test set
- **Production-Ready**: Includes REST API and web frontend for easy deployment
- **Masters-Level Implementation**: Clean architecture, proper documentation, and best practices

### Architecture

**System Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Text                                    │
└───────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  BERT-base-uncased│          │  StyleDistance   │
│   (Fine-tuned)    │          │   (Pre-trained)  │
└─────────┬─────────┘          └─────────┬────────┘
          │                               │
          │ Semantic Embeddings            │ Stylistic Embeddings
          │ (768 dimensions)               │ (768 dimensions)
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Concatenate         │
              │   (1536 dimensions)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  3-Layer MLP           │
              │  512 → 256 → 2         │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Prediction           │
              │  (Human / AI-Generated)│
              └───────────────────────┘
```

**Note:** For a detailed visual architecture diagram, see `docs/architecture.png` (to be created using ioLinks or similar diagramming software).

### Model Components

1. **BERT-base-uncased**: Fine-tuned for binary sequence classification
2. **StyleDistance**: Pre-trained model for extracting stylistic features
3. **Integrated Classifier**: 3-layer feed-forward neural network (512 → 256 → 2)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 705.641-nlp-final-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if needed)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### Running the Application

#### ⚠️ IMPORTANT: Model Setup Required

**The system requires trained models for optimal performance (96.7% accuracy).**

Without trained models, it falls back to pre-trained BERT only (~87% accuracy).

#### Option 1: Using Your Trained Models (Recommended)

If you trained models in the notebooks:

1. **Find your models:**
   ```bash
   python scripts/find_models.py
   ```

2. **Set environment variables:**
   ```bash
   export BERT_MODEL_PATH="path/to/bert-checkpoint"
   export CLASSIFIER_MODEL_PATH="path/to/integrated-classifier.pth"
   ```

3. **Or add to `.env` file:**
   ```bash
   BERT_MODEL_PATH=path/to/bert-checkpoint
   CLASSIFIER_MODEL_PATH=path/to/integrated-classifier.pth
   ```

4. **Run the application:**
   ```bash
   ./run.sh
   ```

#### Option 2: Train Models Locally

If you don't have trained models:

```bash
# First, prepare your data (run data preprocessing notebook)
# Then train:
python train_model.py --data-dir ./data --output-dir ./models
```

#### Option 3: Without Trained Models (Lower Accuracy)

The system will work but with reduced accuracy (~87% vs 96.7%):

```bash
./run.sh
```

**See [MODEL_SETUP.md](MODEL_SETUP.md) for detailed instructions.**

### Accessing the Application

1. **Web Interface**: Open your browser and navigate to `http://localhost:5151`
2. **API Endpoint**: `http://localhost:5151/predict`

---

## 📚 Project Structure

```
705.641-nlp-final-project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py          # MLP classifier architecture
│   └── inference/
│       ├── __init__.py
│       └── predictor.py            # Inference pipeline
├── static/
│   ├── index.html                 # Frontend HTML
│   ├── styles.css                 # Frontend CSS
│   └── app.js                     # Frontend JavaScript
├── baseline_model.ipynb           # Baseline TF-IDF + Logistic Regression
├── nlp_final_proj_data_prep.ipynb # Data preprocessing pipeline
├── nlp_final_proj_model_training.ipynb  # Model training notebook
├── transformer_model.ipynb        # DistilBERT experiment
├── app.py                         # Flask API server
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔬 Model Training

### Data Preparation

1. **Run the data preprocessing notebook** (`nlp_final_proj_data_prep.ipynb`):
   - Filters data (removes "Unknown" sources, text >500 words)
   - Generates StyleDistance embeddings
   - Splits into train/validation/test sets
   - Saves preprocessed data as Parquet files

### Model Training

1. **Fine-tune BERT** (in `nlp_final_proj_model_training.ipynb`):
   - Loads preprocessed data
   - Fine-tunes BERT-base-uncased for binary classification
   - Saves fine-tuned model checkpoint

2. **Extract embeddings**:
   - Uses fine-tuned BERT to extract semantic embeddings
   - Concatenates with StyleDistance embeddings

3. **Train integrated classifier**:
   - Trains 3-layer MLP on concatenated embeddings
   - Saves best model based on validation accuracy

### Expected Results

- **Training Accuracy**: ~96.8%
- **Validation Accuracy**: ~96.8%
- **Test Accuracy**: ~96.7%
- **F1-Score**: ~0.97

---

## 🌐 API Documentation

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Text Classifier"
}
```

### Single Text Prediction

```http
POST /predict
Content-Type: application/json

{
  "text": "Your text to analyze here..."
}
```

**Response:**
```json
{
  "label": "AI-Generated",
  "is_ai": true,
  "confidence": 0.95,
  "probabilities": {
    "human": 0.05,
    "ai_generated": 0.95
  }
}
```

### Batch Prediction

```http
POST /predict/batch
Content-Type: application/json

{
  "texts": ["text1", "text2", "text3"]
}
```

**Response:**
```json
{
  "results": [
    {
      "label": "Human",
      "is_ai": false,
      "confidence": 0.92,
      "probabilities": {
        "human": 0.92,
        "ai_generated": 0.08
      }
    },
    ...
  ]
}
```

### Error Responses

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing/invalid input)
- `500`: Internal server error

---

## 🧪 Usage Examples

### Python API Client

```python
import requests

# Single prediction
response = requests.post('http://localhost:5151/predict', json={
    'text': 'This is a sample text to analyze.'
})
result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
response = requests.post('http://localhost:5151/predict/batch', json={
    'texts': ['Text 1', 'Text 2', 'Text 3']
})
results = response.json()['results']
```

### Command Line (using curl)

```bash
curl -X POST http://localhost:5151/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

---

## 📊 Performance Metrics

### Test Set Results

- **Accuracy**: 96.74%
- **Precision**: 97.80%
- **Recall**: 96.88%
- **F1-Score**: 97.34%

### Baseline Comparison

- **Baseline (TF-IDF + Logistic Regression)**: 87.5% accuracy
- **Our Model (BERT + StyleDistance + MLP)**: 96.7% accuracy
- **Improvement**: +9.2% absolute improvement

---

## 🔍 Technical Details

### Model Architecture

**Integrated Classifier (MLP)**:
- Input: 1536-dimensional concatenated embeddings
- Layer 1: 512 units + ReLU + Dropout(0.3)
- Layer 2: 256 units + ReLU + Dropout(0.3)
- Output: 2 units (binary classification)

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Epochs**: 10
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Based on validation accuracy

### Feature Dimensions

- **BERT Embeddings**: 768 dimensions
- **StyleDistance Embeddings**: 768 dimensions
- **Integrated Embeddings**: 1536 dimensions

---

## 📖 References

1. R. A. R. Soto, K. Koch, A. Khan, B. Chen, M. Bishop, and N. Andrews, "Few-Shot Detection of Machine-Generated Text using Style Representations," Lawrence Livermore National Laboratory/Johns Hopkins University Technical Report, 2024.

2. R. A. R. Soto, B. Chen, and N. Andrews, "Language Models Optimized to Fool Detectors Still Have a Distinct Style (And How to Change It)," arXiv e-print arXiv:2505.14608, May 2025.

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

---

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

---

## 📝 License

This project is for educational purposes as part of EN 705.641 NLP course.

---

## 🤝 Contributing

This is a final project submission. For questions or issues, please contact the project authors.

---

## 🙏 Acknowledgments

- Hugging Face for transformer models and libraries
- StyleDistance team for the stylistic embedding model
- Kaggle for the Human Vs. LLM Text Corpus dataset
