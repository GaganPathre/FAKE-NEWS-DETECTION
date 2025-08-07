📰 **FAKE-NEWS-DETECTION**

This repository contains the complete implementation, results, and documentation for a Fake News Detection system using both classical machine learning techniques and a deep learning-based Three-Level Hierarchical Attention Network (3HAN). The project was developed as part of a Bachelor's dissertation.


🔍 **Project Overview**

The goal of this project is to automatically detect fake news articles using textual data collected from various real-world datasets. It involves a two-phased experimental approach:
Baseline Models: Logistic Regression and Random Forest classifiers trained on engineered features.
Advanced Model: A custom-built 3HAN model using a hierarchical attention mechanism designed to capture word-, sentence-, and document-level semantics.


📊 **Datasets Used**

LIAR dataset

ISOT Fake News Dataset

WeLFake

FakeNewsNet

BuzzFeed News & PolitiFact


🛠️ **Technologies & Libraries**

Python 3.10

PyTorch

Scikit-learn

NLTK

NumPy, pandas

Matplotlib, Seaborn

Google Colab (T4 GPU)


📈 **Evaluation Metrics**

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC


🧠 **Key Features**

Data cleaning and preprocessing pipeline

Feature extraction with TF-IDF and n-gram analysis

Deep learning implementation using 3HAN with attention layers

Visualization of attention weights for model interpretability

Comparison between traditional ML and deep learning approaches

Model evaluation with statistical testing and error analysis


📌 Future Work

Extend model to multilingual fake news detection

Integrate real-time data streams from Telegram/WhatsApp

Explore hybrid models using explainable AI (XAI) principles
