🛒 Online Shopper Purchase Intention Prediction
📌 Overview

This project predicts whether an online shopper will make a purchase during a browsing session using behavioral and session-based data. By analyzing user interactions such as page visits, time spent, and bounce rates, the system identifies purchase intent in real time.

The project applies both classical machine learning and deep learning models and compares their performance to highlight trade-offs between accuracy and interpretability.

🎯 Objectives

Predict online purchase intent

Analyze customer browsing behavior

Compare traditional ML and deep learning models

Provide interpretable insights for e-commerce optimization

Ensure reproducibility using a synthetic data pipeline

🧠 Models Used

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Multilayer Perceptron (MLP)

📊 Dataset

Online Shoppers Purchasing Intention Dataset (UCI)

Includes an automatic synthetic data generator for reproducible experiments when real data is unavailable

🔄 Workflow

Data preprocessing & feature engineering

Model training

Performance evaluation

Model comparison

Feature importance analysis

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

🔍 Key Findings

Ensemble models (Random Forest, XGBoost) outperform traditional classifiers

MLP provides competitive results with higher computational cost

Feature importance reveals key behavioral drivers of purchase decisions

⚙️ Tech Stack

Python

NumPy, Pandas

Scikit-learn

XGBoost

TensorFlow / Keras

Matplotlib, Seaborn

🚀 Use Cases

Conversion rate optimization

Personalized marketing

Customer behavior analysis

Real-time intent prediction
