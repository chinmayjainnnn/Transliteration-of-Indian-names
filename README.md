# Modeling Indian First Names Using Character-Level Language Models

This project aims to model Indian first names using character-level language models. By leveraging the inherent patterns in Indian names, we start by modeling them using n-gram models and then move to neural n-gram and Recurrent Neural Network (RNN) models.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Description](#description)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [N-gram Models](#n-gram-models)
    - [Unigram Model](#unigram-model)
    - [Bigram Model](#bigram-model)
    - [Trigram Model](#trigram-model)
  - [Neural N-gram Model](#neural-n-gram-model)
  - [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
- [Results](#results)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Models](#running-the-models)

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- TorchText
- NumPy
- Pandas
- Matplotlib


```
torch>=1.7.0
torchtext
numpy
pandas
matplotlib
```

## Dataset

The datasets used in this project are:

- **Training Data**: `train_data.csv`
- **Validation Data**: `valid_data.csv`
- **Evaluation Prefixes**: `eval_prefixes.txt`
- **Evaluation Sequences**: `eval_sequences.txt`

### Downloading the Dataset

You can download the datasets using the following commands:

```bash
wget -O train_data.csv "https://docs.google.com/spreadsheets/d/1AUzwOQQbAehg_eoAMCcWfwSGhKwSAtnIzapt2wbv0Zs/gviz/tq?tqx=out:csv&sheet=train_data.csv"
wget -O valid_data.csv "https://docs.google.com/spreadsheets/d/1UtQErvMS-vcQEwjZIjLFnDXlRZPxgO1CU3PF-JYQKvA/gviz/tq?tqx=out:csv&sheet=valid_data.csv"
wget -O eval_prefixes.txt "https://drive.google.com/uc?export=download&id=1tuRLJXLd2VcDaWENr8JTZMcjFlwyRo60"
wget -O eval_sequences.txt "https://drive.google.com/uc?export=download&id=1kjPAR04UTKmdtV-FJ9SmDlotkt-IKM3b"
```

```bash
bash download_data.sh
```


## Description

Indian names possess unique patterns and structures that can be learned using language models. This project explores various models to capture these patterns and generate new names that resemble authentic Indian first names.

## Methodology

### Data Preprocessing

- **Tokenization**: Names are tokenized into individual characters.
- **Vocabulary Building**: A character-level vocabulary is built from the dataset.
- **Special Tokens**:
  - `<s>`: Start-of-name token.
  - `</s>`: End-of-name token.
  - `<unk>`: Unknown character token for out-of-vocabulary characters.

### N-gram Models

An n-gram model predicts the next character based on the previous `n-1` characters.

#### Unigram Model

- **Definition**: Considers each character independently.
- **Probability Estimation**:
  $$ P(c) = \frac{\text{Count}(c)}{\sum_{c'} \text{Count}(c')} $$
- **Smoothing**: Add-1 (Laplace) smoothing is applied to handle zero probabilities.

#### Bigram Model

- **Definition**: Considers the probability of a character given the previous character.
- **Probability Estimation**:
  $$ P(c_i | c_{i-1}) = \frac{\text{Count}(c_{i-1}, c_i)}{\sum_{c'} \text{Count}(c_{i-1}, c')} $$
- **Smoothing Techniques**:
  - **Add-k Smoothing**: A constant `k` is added to counts to prevent zero probabilities.
  - **Interpolation**: Combines unigram and bigram probabilities:
    $$ P_{\text{interpolated}} = \lambda_1 P_{\text{bigram}} + \lambda_2 P_{\text{unigram}} $$
  - **Hyperparameters**:
    - **Add-k**: `k = 0.5`
    - **Lambdas**: `(λ1, λ2) = (0.7, 0.3)`

#### Trigram Model

- **Definition**: Considers the probability of a character given the two previous characters.
- **Smoothing**: Interpolation smoothing combining unigram, bigram, and trigram probabilities.
  $$ P_{\text{interpolated}} = \lambda_1 P_{\text{trigram}} + \lambda_2 P_{\text{bigram}} + \lambda_3 P_{\text{unigram}} $$
- **Hyperparameters**:
  - **Lambdas**: `(λ1, λ2, λ3) = (0.6, 0.25, 0.15)`

### Neural N-gram Model

A Feedforward Neural Network (FNN) is implemented to model character sequences.

- **Architecture**:
  - **Embedding Layer**: Transforms characters into dense vector representations.
  - **Hidden Layer**: Applies non-linear transformation using the `tanh` activation function.
  - **Output Layer**: Produces a probability distribution over the vocabulary.
- **Equation**:
  $$ y = b + W x + U \tanh(d + H x) $$
- **Training Details**:
  - **Embedding Size**: `64`
  - **Hidden Layer Size**: `256`
  - **Optimizer**: Adam optimizer with a learning rate of `0.001`.
  - **Epochs**: `5`
  - **Batch Size**: `128`

### Recurrent Neural Network (RNN)

An RNN is proposed to model longer dependencies in names. However, the implementation is incomplete and requires further development.

## Results

### Perplexity Scores

- **Unigram Model**:
  - **Train Perplexity**: Variable (depends on smoothing).
- **Bigram Model**:
  - **Train Perplexity**: Variable (depends on smoothing and hyperparameters).
- **Trigram Model**:
  - **Train Perplexity**: Variable (depends on smoothing and hyperparameters).
- **Neural N-gram Model**:
  - **Train Perplexity**: Variable (based on training).

*(Replace "Variable" with actual perplexity scores from your experiments.)*

## Usage

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/indian-name-language-model.git
cd indian-name-language-model
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download the dataset**:

```bash
bash download_data.sh
```

### Running the Models

1. **Unigram Model**:

```bash
python unigram_model.py
```

2. **Bigram Model**:

```bash
python bigram_model.py
```

3. **Trigram Model**:

```bash
python trigram_model.py
```

4. **Neural N-gram Model**:

```bash
python neural_ngram_model.py
```

*(Ensure that each model's script is properly set up and available in the repository.)*

### Directory Structure

```
indian-name-language-model/
├── SAPname_SRno_assignment2.py
├── SAPname_SRno/
│   ├── fnn/
│   │   ├── model.pt
│   │   ├── vocab.pt
│   │   └── loss.json
│   └── rnn/
│       ├── model.pt
│       ├── vocab.pt
│       └── loss.json
├── data/
│   ├── train_data.csv
│   ├── valid_data.csv
│   ├── eval_prefixes.txt
│   └── eval_sequences.txt
├── download_data.sh
├── requirements.txt
└── README.md
```

*(Replace `SAPname_SRno` with your actual SAP name and SR number.)*

## Example Results

### Generated Names

**Unigram Model** (without smoothing):

- `awezuou`, `zedj]cs`, `jb=tw|o`, `pnwkzd#`, `m#}puwg`

**Bigram Model** (with interpolation):

- `aaroo`, `aaley`, `aama`, `aaria`, `aaka`

**Trigram Model** (with interpolation):

- `aaradhana`, `aarathi`, `aarti`, `aarushi`, `aaryan`

**Neural N-gram Model**:

- `aarav`, `aanya`, `aakash`, `aashish`, `aaryan`

### Most Likely Characters After Sequence

For the sequence `'aar'`:

- **Bigram Model**: `a`, `i`, `o`, `u`, `e`
- **Trigram Model**: `a`, `i`, `u`, `e`, `t`
- **Neural N-gram Model**: `a`, `i`, `o`, `u`, `e`

### Perplexity Comparison

- **Unigram Model (Unsmoothened)**: Infinite (due to zero probabilities)
- **Unigram Model (Add-1 Smoothing)**: Lower perplexity, better generalization
- **Bigram Model**: Lower perplexity than unigram models
- **Trigram Model**: Lower perplexity than bigram models
- **Neural N-gram Model**: Comparable or better perplexity than trigram model

## References

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). *Journal of Machine Learning Research*, 3(Feb), 1137-1155.
- Jurafsky, D., & Martin, J. H. (2020). [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/). Chapter 3: N-gram Language Models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

