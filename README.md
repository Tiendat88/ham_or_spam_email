# Spam Email Detection Project

A comprehensive machine learning project that implements and compares three different approaches for email spam detection: Random Forest, Logistic Regression, and LSTM (Long Short-Term Memory) neural networks.

## Project Overview

This project builds and evaluates multiple machine learning models to classify emails as spam or legitimate (ham). The implementation includes data preprocessing, feature extraction using TF-IDF, and deep learning with LSTM networks.

## Dataset

The project uses a combined dataset (`combined_data.csv`) with:
- **Total samples**: 83,448 emails
- **Features**: Text content and binary labels (0 = not spam, 1 = spam)
- **Distribution**: 
  - Spam emails: 43,910 (52.6%)
  - Not spam emails: 39,538 (47.4%)

## Models Implemented

### 1. Random Forest Classifier
- **Accuracy**: 98.54%
- **Precision**: 98.40%
- **Recall**: 98.82%
- **F1-Score**: 98.61%

### 2. Logistic Regression
- **Accuracy**: 98.35%
- **Precision**: 98.03%
- **Recall**: 98.83%
- **F1-Score**: 98.43%

### 3. LSTM Neural Network
- **Accuracy**: 97.97%
- **Precision**: 98.42%
- **Recall**: 97.69%
- **F1-Score**: 98.06%

## Features

- **Text Preprocessing**: Lowercasing, punctuation removal, stopword filtering
- **Feature Extraction**: TF-IDF vectorization with 5,000 max features
- **Deep Learning**: LSTM with embedding layer, dropout for regularization
- **Model Comparison**: Comprehensive evaluation metrics for all models
- **Prediction Pipeline**: Ready-to-use prediction functions for new emails

## Project Structure

```
spam-detection/
│
├── A39948_nlp.ipynb           # Main notebook with all implementations
├── combined_data.csv          # Dataset file
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── models/                    # Saved model files
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── lstm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer.pkl
└── predictions/               # Sample predictions and results
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spam-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Running the Complete Pipeline

Open and run the Jupyter notebook `A39948_nlp.ipynb` to:

1. Load and explore the dataset
2. Preprocess the text data
3. Train all three models
4. Evaluate and compare performance
5. Save trained models
6. Test predictions on sample emails

### Using Trained Models

```python
import joblib
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
lstm_model = joblib.load('lstm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
tokenizer = joblib.load('tokenizer.pkl')

# Preprocess new email
def preprocess_text(text):
    # Your preprocessing function here
    pass

# Predict with Random Forest/Logistic Regression
new_email = "Your email text here"
processed_email = preprocess_text(new_email)
email_tfidf = vectorizer.transform([processed_email])
prediction = rf_model.predict(email_tfidf)

# Predict with LSTM
email_sequence = tokenizer.texts_to_sequences([processed_email])
email_padded = pad_sequences(email_sequence, maxlen=100, padding='post')
lstm_prediction = (lstm_model.predict(email_padded) > 0.5).astype(int)
```

## Model Architecture

### LSTM Model Details
- **Embedding Layer**: 5,000 vocabulary size, 64-dimensional embeddings
- **LSTM Layer**: 64 units, single direction
- **Dropout**: 0.5 for regularization
- **Output Layer**: Single sigmoid neuron for binary classification
- **Optimization**: Adam optimizer with binary crossentropy loss
- **Early Stopping**: Patience of 3 epochs to prevent overfitting

### Traditional ML Models
- **TF-IDF Vectorization**: Maximum 5,000 features
- **Random Forest**: 100 estimators
- **Logistic Regression**: Maximum 1,000 iterations

## Performance Analysis

All three models achieve excellent performance with accuracy above 97%. The Random Forest classifier shows the best overall performance, followed closely by Logistic Regression. The LSTM model, while slightly lower in accuracy, demonstrates the power of deep learning for text classification tasks.

## Sample Predictions

The trained models successfully classify:
- **Spam**: "Congratulations! You've won a $1000 Walmart gift card..."
- **Not Spam**: "Hi John, I hope you are doing well. Let's schedule a meeting..."

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers for the email spam classification data
- TensorFlow and Scikit-learn communities for excellent ML libraries
- NLTK team for natural language processing tools

## Future Improvements

- Implement more advanced NLP techniques (BERT, transformers)
- Add real-time email classification API
- Enhance preprocessing with advanced text cleaning
- Implement ensemble methods combining all three approaches
- Add support for multilingual spam detection
