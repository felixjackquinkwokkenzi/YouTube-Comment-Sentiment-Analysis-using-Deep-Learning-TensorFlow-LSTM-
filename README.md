# YouTube Comment Sentiment Analysis (Deep Learning – TensorFlow)

This repository contains an end-to-end sentiment analysis pipeline for YouTube comments.  
The workflow includes:

- scraping comments using the official YouTube Data API
- text preprocessing (regex cleaning, stopword removal, stemming, lemmatization)
- auto sentiment labeling using NLTK VADER
- sequence modeling using a Bidirectional LSTM
- evaluation + inference / testing

This project is one of the final projects I completed during Dicoding’s **Belajar Fundamental Deep Learning** program.

---

## Features

- YouTube comment scraping via Google API
- Data cleaning + normalization + tokenization
- Sentiment labeling using VADER (positive / negative / neutral)
- Bidirectional LSTM classifier (end-to-end deep learning pipeline)
- Training / evaluation + accuracy & classification report
- Inference for custom sentences

---

## Dataset

The raw text is scraped directly from public YouTube video comments using the YouTube Data API v3.

Example extraction:

```python
comments = get_video_comments(video_id, 10000)
```

Resulting data is stored into CSV:

```python
df.to_csv('youtube_comments.csv', index=False)
```

Later processed + labeled version is saved as:

```python
youtube_comments_labeled.csv
```

---

## Model Architecture

| Stage          | Technique                                  |
| -------------- | ------------------------------------------ |
| Tokenization   | Keras Tokenizer                            |
| Embedding      | 128-dim embedding layer                    |
| Sequence Model | Bidirectional LSTM (return_sequences=True) |
| Pooling        | GlobalMaxPool1D                            |
| Dense          | 128 Relu + Dropout                         |
| Output         | 3-class Softmax                            |

Loss: sparse_categorical_crossentropy
Optimizer: Adam (1e-4)

---

## Training

```python
history = model.fit(
    train_padded,
    train_labels,
    epochs=10,
    verbose=1,
    validation_split=0.1,
    batch_size=64
)
```

---

## Evaluation

```python
pred = model.predict(test_padded)
print(accuracy_score(test_labels, pred_labels))
print(classification_report(test_labels, pred_labels, target_names=encoder.classes_))
```

---

## Prediction Example

```python
sentence = ["This community is very helpful!"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length)
prediction = model.predict(padded)
```

---

## Author

This project was created by Felix Jackquin Kwok Kenzi.

This project is one of the final projects completed as part of the 'Belajar Fundamental Deep Learning' program from Dicoding.com.
