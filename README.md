# DIET Classifier PyTorch

A PyTorch implementation of the **DIET (Dual Intent and Entity Transformer)** architecture for joint intent classification and named entity recognition in dialogue systems.

This implementation is based on the following papers:
- [DIET: Lightweight Language Understanding for Dialogue Systems](https://arxiv.org/abs/2004.09936) (Bunk et al., 2020)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) (Lample et al., 2016)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
  - [Architecture Diagram](#architecture-diagram)
  - [Component Description](#component-description)
- [Mathematical Formulation](#mathematical-formulation)
  - [Sparse Feature Extraction](#sparse-feature-extraction)
  - [Transformer Encoder](#transformer-encoder)
  - [Conditional Random Field (CRF)](#conditional-random-field-crf)
- [Training Pipeline](#training-pipeline)
  - [DataLoader](#dataloader)
  - [Trainer](#trainer)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

DIET is a lightweight multi-task architecture designed for Natural Language Understanding (NLU) in dialogue systems. It jointly predicts:

1. **Intent Classification**: Determining the user's intention (e.g., "turn_on_light", "turn_off_plug")
2. **Entity Recognition**: Extracting relevant entities using BIO tagging (e.g., room names, device names)

The key advantage of DIET is that it achieves competitive performance without requiring large pre-trained language models, making it efficient for production deployment.

---

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. To install the project with uv:

```bash
# Clone the repository
git clone https://github.com/opfernandez/diet_classifier_pytorch.git
cd diet_classifier_pytorch

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate 

# Install the package in development mode
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/opfernandez/diet_classifier_pytorch.git
cd diet_classifier_pytorch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install the package
pip install -e .
```

### Dependencies

The project requires the following packages:
- `torch` - PyTorch deep learning framework
- `pyyaml` - YAML file parsing
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing

---

## Architecture

### Architecture Diagram

```
                              Input Text
                                  |
                                  v
                    +---------------------------+
                    |       Tokenization        |
                    |   (Whitespace + [CLS])    |
                    +---------------------------+
                                  |
                                  v
         +------------------------------------------------+
         |           Sparse Feature Extraction            |
         |                                                |
         |  +------------------+  +--------------------+  |
         |  |   Word Lookup    |  |  Char N-gram Hash  |  |
         |  |   (Embedding)    |  |  (EmbeddingBag)    |  |
         |  +------------------+  +--------------------+  |
         |            |                   |               |
         |            +-------+   +-------+               |
         |                    |   |                       |
         |                    v   v                       |
         |              +-------------+                   |
         |              |     Add     |                   |
         |              +-------------+                   |
         +------------------------------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   Feed-Forward + ReLU     |
                    |      + LayerNorm          |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   Transformer Encoder     |
                    |      (N layers)           |
                    |   - Self-Attention        |
                    |   - Feed-Forward          |
                    +---------------------------+
                                  |
                +----------------+----------------+
                |                                 |
                v                                 v
    +---------------------+           +----------------------+
    |   Entity Head       |           |    Intent Head       |
    | (Linear + CRF)      |           | (Linear + Softmax)   |
    |                     |           |                      |
    | Input: All tokens   |           | Input: [CLS] token   |
    +---------------------+           +----------------------+
                |                                 |
                v                                 v
        Entity Tags                      Intent Prediction
       (BIO format)                     (Class probabilities)
```

### Component Description

#### 1. Sparse Feature Extractor (`sparse_features_extractor.py`)

The sparse feature extractor converts raw text into dense representations through two parallel mechanisms:

- **Word Embeddings**: Each token is mapped to a learned embedding vector using a vocabulary dictionary. Out-of-vocabulary words are mapped to a special `[UNK]` token.

- **Character N-gram Embeddings**: For each token, character n-grams (default: 2 to 5 characters) are extracted and combined using an `EmbeddingBag` layer. This provides sub-word information and handles morphological variations. N-grams not in the dictionary are hashed to an overflow area using deterministic hashing.

The final token representation is the sum of word and n-gram embeddings:

$$\mathbf{h}_t = \mathbf{e}_{word}(w_t) + \sum_{g \in ngrams(w_t)} \mathbf{e}_{ngram}(g)$$

#### 2. Transformer Encoder

A standard Transformer encoder stack processes the token representations:

- **Multi-Head Self-Attention**: Allows each token to attend to all other tokens in the sequence
- **Position-wise Feed-Forward Networks**: Two linear transformations with ReLU activation
- **Layer Normalization and Dropout**: Applied for regularization

The encoder captures contextual relationships between tokens, essential for both entity recognition and intent understanding.

#### 3. Intent Classification Head

The intent is predicted using only the `[CLS]` token representation:

$$\mathbf{y}_{intent} = \text{softmax}(\mathbf{W}_{intent} \cdot \mathbf{h}_{[CLS]} + \mathbf{b}_{intent})$$

The loss is computed using Cross-Entropy:

$$\mathcal{L}_{intent} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

#### 4. Conditional Random Field (CRF) for Entity Recognition (`crf.py`)

For sequence labeling, a linear-chain CRF is applied on top of the Transformer outputs. The CRF models dependencies between adjacent labels, which is crucial for BIO tagging consistency.

---

## Mathematical Formulation

### Conditional Random Field (CRF)

Following Lample et al. (2016), the CRF defines a probability distribution over tag sequences.

#### Score Function

For an input sequence $\mathbf{x}$ and tag sequence $\mathbf{y}$, the score is:

$$s(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^{T} \left( A_{y_{t-1}, y_t} + P_{t, y_t} \right)$$

Where:
- $A \in \mathbb{R}^{K \times K}$ is the transition matrix (learned parameter)
- $P \in \mathbb{R}^{T \times K}$ are the emission scores from the neural network
- $A_{i,j}$ represents the score of transitioning from tag $i$ to tag $j$

#### Probability

The probability of a tag sequence is normalized over all possible sequences:

$$p(\mathbf{y}|\mathbf{x}) = \frac{\exp(s(\mathbf{x}, \mathbf{y}))}{\sum_{\tilde{\mathbf{y}} \in \mathcal{Y}_\mathbf{x}} \exp(s(\mathbf{x}, \tilde{\mathbf{y}}))}$$

#### Training Loss

The CRF loss is the negative log-likelihood:

$$\mathcal{L}_{CRF} = -\log p(\mathbf{y}|\mathbf{x}) = \log Z(\mathbf{x}) - s(\mathbf{x}, \mathbf{y})$$

Where $Z(\mathbf{x}) = \sum_{\tilde{\mathbf{y}}} \exp(s(\mathbf{x}, \tilde{\mathbf{y}}))$ is the partition function computed efficiently using the forward algorithm.

#### Forward Algorithm

The partition function is computed in $O(T \cdot K^2)$ time using dynamic programming:

$$\alpha_t(j) = \sum_{i=1}^{K} \alpha_{t-1}(i) \cdot \exp(A_{i,j} + P_{t,j})$$

In log-space (for numerical stability):

$$\log \alpha_t(j) = \text{logsumexp}_i \left( \log \alpha_{t-1}(i) + A_{i,j} + P_{t,j} \right)$$

#### Viterbi Decoding

At inference time, the most likely tag sequence is found using the Viterbi algorithm:

$$\mathbf{y}^* = \arg\max_{\mathbf{y}} s(\mathbf{x}, \mathbf{y})$$

#### Total Training Loss

The combined loss for joint training:

$$\mathcal{L}_{total} = \mathcal{L}_{CRF} + \mathcal{L}_{intent}$$

---

## Training Pipeline

### DataLoader (`data_loader.py`)

The `DataLoader` class handles data loading and preprocessing for the DIET model.

**Responsibilities:**

1. **YAML Parsing**: Reads training data in Rasa NLU format from YAML files
2. **Entity Extraction**: Parses bracketed entity annotations (e.g., `[salon](sala)`) and converts them to BIO tags
3. **Tokenization**: Splits text by whitespace and prepends `[CLS]` token
4. **Padding**: Aligns sequences within a batch to the same length using `[PAD]` tokens
5. **Batch Creation**: Groups samples into batches and shuffles them randomly

**Input Format** (Rasa NLU YAML):

```yaml
nlu:
- intent: turn_on_light
  examples: |
    - turn on the light in the [kitchen](room)
    - please activate the [living room](room) lights
```

**Output Format**:

```python
{
    "text": "[CLS] turn on the light in the kitchen",
    "entity_tags": ["O", "O", "O", "O", "O", "O", "O", "B-room"],
    "intent": "turn_on_light"
}
```

**Key Method - `format_batch()`**:
- Converts a list of samples into tensors ready for the model
- Returns: `(text_inputs, entity_tag_indices, one_hot_intent_labels)`

### Trainer (`trainer.py`)

The `Trainer` class orchestrates the training loop and model checkpointing.

**Responsibilities:**

1. **Optimization**: Uses AdamW optimizer with configurable learning rate
2. **Training Loop**: Iterates over epochs and batches, computing gradients
3. **Checkpointing**: Saves model weights when validation loss improves
4. **Visualization**: Plots training loss history using matplotlib

**Training Step**:

```
For each batch:
    1. Format batch data (text, entity labels, intent labels)
    2. Move tensors to device (CPU/GPU)
    3. Zero gradients
    4. Forward pass through model (train_forward)
    5. Compute combined loss (CRF + CrossEntropy)
    6. Backward pass
    7. Update weights
```

**Hyperparameters** (configurable in `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Number of samples per batch |
| `lr` | 1e-3 | Learning rate for AdamW |
| `epochs` | 20 | Number of training epochs |
| `device` | auto | 'cuda' if available, else 'cpu' |

---

## Usage

### Training

```bash
cd scripts
python train.py
```

This will:
1. Load training data from `data/data.yml`
2. Build word and n-gram dictionaries
3. Train the DIET model
4. Save the best checkpoint to `models/diet_model.pt`
5. Display a loss history plot

### Validation

```bash
cd scripts
python validation.py
```

This will:
1. Load validation data from `data/validation.yml`
2. Load the trained model from `models/diet_model.pt`
3. Compute F1 score for entity recognition
4. Compute accuracy for intent classification
5. Display confusion matrices

### Inference Server

The project includes a socket-based inference server for production deployment:

```bash
python -m diet_classifier.inference.server
```

This starts a TCP server on `0.0.0.0:5555` that accepts JSON requests:

```bash
# Example client request
echo '{"text": "encender la luz de la cocina"}' | nc localhost 5555
```

**Response format:**
```json
{
    "status": "success",
    "result": {
        "text": "encender la luz de la cocina",
        "intent": "turn_on_light",
        "intent_confidence": 0.95,
        "entities": [
            {"type": "room", "start": 4, "end": 4, "words": "cocina"}
        ],
        "inference_time_ms": 2.5
    }
}
```

### Programmatic Inference

```python
from diet_classifier.inference import DIETServer

# Initialize server (also works for direct inference)
server = DIETServer(
    device='cuda',  # or 'cpu'
    model_path='models/diet_model.pt',
    word_dict_path='data/word_dict.json',
    ngram_dict_path='data/ngram_dict.json',
    entity_labels_path='data/entity_labels.json',
    intent_labels_path='data/intent_labels.json'
)

# Perform inference
results = server.predict(["encender la luz de la cocina"])
print(results[0])
```

---

## Project Structure

```
diet_classifier_pytorch/
├── pyproject.toml              # Project configuration and dependencies
├── README.md
│
├── data/
│   ├── data.yml                # Training data in Rasa NLU format
│   ├── validation.yml          # Validation data
│   ├── word_dict.json          # Generated word vocabulary
│   ├── ngram_dict.json         # Generated n-gram vocabulary
│   ├── entity_labels.json      # Entity label definitions (BIO tags)
│   └── intent_labels.json      # Intent label definitions
│
├── models/
│   └── diet_model.pt           # Trained model weights
│
├── scripts/
│   ├── train.py                # Training entry point
│   └── validation.py           # Validation and metrics computation
│
└── src/
    └── diet_classifier/
        ├── __init__.py
        │
        ├── model/
        │   ├── __init__.py
        │   ├── diet.py                     # Main DIET model architecture
        │   ├── crf.py                      # Conditional Random Field implementation
        │   └── sparse_features_extractor.py # Word and n-gram feature extraction
        │
        ├── training/
        │   ├── __init__.py
        │   ├── trainer.py                  # Training loop and checkpointing
        │   └── data_loader.py              # Data loading and preprocessing
        │
        └── inference/
            ├── __init__.py
            └── server.py                   # Socket-based inference server
```

---

## References

1. Bunk, T., Varshneya, D., Vlasov, V., & Nichol, A. (2020). *DIET: Lightweight Language Understanding for Dialogue Systems*. arXiv:2004.09936.

2. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). *Neural Architectures for Named Entity Recognition*. NAACL-HLT 2016. arXiv:1603.01360.

---

## License

This project is provided as-is for educational and research purposes.
