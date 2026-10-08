# Twitter Sentiment Analysis (DistilBERT + Flask)

Fine-tunes **DistilBERT** on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset (1.6M tweets) to classify English text as **positive** or **negative**, and serves the model through a small **Flask** web app.

## Features

- DistilBERT (`distilbert-base-uncased`) fine-tuned for binary classification
- Trains on a 300k-tweet sample with an 80/20 stratified split
- Class-weighted loss with label smoothing, gradient clipping and a linear LR scheduler
- Early stopping on validation F1; the best model is saved to `distilbert_sentiment.pt`
- Runs on GPU when available, falls back to CPU
- Simple web UI: type a sentence, get `Pozitif` / `Negatif`

## Tech stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · Flask

## Project structure

```
├── sentiment.py          # dataset, training loop, evaluation, prediction
├── app.py                # Flask web app that loads the trained model
├── templates/index.html  # web UI
└── requirements.txt
```

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and put `training.1600000.processed.noemoticon.csv` in the project folder.

3. **Train the model** (creates `distilbert_sentiment.pt`):

   ```bash
   python sentiment.py
   ```

   Settings such as `SAMPLE_SIZE`, `EPOCHS` and `BATCH_SIZE` are at the top of `sentiment.py`. A GPU is strongly recommended.

4. **Run the web app**

   ```bash
   python app.py
   ```

   Open http://127.0.0.1:5000 in your browser.

## Example

| Input | Prediction |
|---|---|
| I absolutely love this new phone! | Pozitif |
| This movie was a complete waste of time and money. | Negatif |
