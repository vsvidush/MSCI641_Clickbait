import json
import re
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Stop words
stop_words = set(stopwords.words('english'))
# Special characters to remove
special_chars = re.compile(r'[' + re.escape('!"#$%&()*+/:;<=>@^`|~\t[]{}\\.-') + r']')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters
    text = special_chars.sub(' ', text)
    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def load_data(path, is_test=False):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if is_test or (isinstance(entry.get('tags'), list) and entry['tags'][0] in {'phrase', 'multi', 'passage'}):
                data.append(entry)
    return data

def prepare_data(data):
    texts = []
    labels = []
    for entry in data:
        postText = preprocess_text(entry.get('postText', [""])[0])
        targetParagraphs = preprocess_text(" ".join(entry.get('targetParagraphs', [])))
        targetTitle = preprocess_text(entry.get('targetTitle', ""))
        targetDescription = preprocess_text(entry.get('targetDescription', ""))
        targetKeywords = preprocess_text(entry.get('targetKeywords', ""))
        humanSpoiler = preprocess_text(entry.get('provenance', {}).get('humanSpoiler', ""))
        spoiler = preprocess_text(" ".join(entry.get('spoiler', [])))

        combined_text = " ".join([postText, targetParagraphs, targetTitle, targetDescription, targetKeywords, humanSpoiler, spoiler])
        texts.append(combined_text)
        
        if 'tags' in entry:
            labels.append(entry['tags'][0])
    
    return texts, labels

# Load Data
train_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\train.jsonl'
val_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\val.jsonl'
test_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\test.jsonl'
output_file_path = 'predictions_SVM_SBERT.csv'  # File to save predictions
performance_file_path = 'performance_metrics_SVM_SBERT.csv'  # File to save performance metrics

train_data = load_data(train_data_path)
val_data = load_data(val_data_path)
test_data = load_data(test_data_path, is_test=True)

# Prepare Data
train_texts, train_labels = prepare_data(train_data)
val_texts, val_labels = prepare_data(val_data)
test_texts, _ = prepare_data(test_data)

# Label encoding
label_encoder = LabelEncoder()
all_labels = train_labels + val_labels
label_encoder.fit(all_labels)
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Load SBERT model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate SBERT embeddings
def encode_texts(texts, model):
    return model.encode(texts, show_progress_bar=True)

train_embeddings = encode_texts(train_texts, sbert_model)
val_embeddings = encode_texts(val_texts, sbert_model)
test_embeddings = encode_texts(test_texts, sbert_model)

# Combine training and validation data for SVM training
X = np.vstack([train_embeddings, val_embeddings])
y = np.hstack([train_labels, val_labels])

# Create SVM model pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC(kernel='linear', probability=True, random_state=42))
])

# Train the SVM model
svm_pipeline.fit(X, y)

# Evaluate the model
val_predictions = svm_pipeline.predict(val_embeddings)

# Calculate metrics
report = classification_report(val_labels, val_predictions, target_names=label_encoder.classes_, output_dict=True)
mcc = matthews_corrcoef(val_labels, val_predictions)
accuracy = accuracy_score(val_labels, val_predictions)

# Print the classification report
print(classification_report(val_labels, val_predictions, target_names=label_encoder.classes_))

# Ensure all arrays are of the same length
num_classes = len(label_encoder.classes_)
performance_metrics = {
    'Class': list(report.keys())[:-3] + ['weighted avg'],
    'Precision': [report[label]['precision'] for label in report if label not in ('accuracy', 'macro avg', 'weighted avg')] + [report['weighted avg']['precision']],
    'Recall': [report[label]['recall'] for label in report if label not in ('accuracy', 'macro avg', 'weighted avg')] + [report['weighted avg']['recall']],
    'F1-Score': [report[label]['f1-score'] for label in report if label not in ('accuracy', 'macro avg', 'weighted avg')] + [report['weighted avg']['f1-score']],
    'MCC': [mcc] * (num_classes + 1),  # same length as number of classes + 1 for weighted avg
    'Accuracy': [accuracy] * (num_classes + 1)  # same length as number of classes + 1 for weighted avg
}

# Debugging lengths
print(f"Lengths: Class={len(performance_metrics['Class'])}, Precision={len(performance_metrics['Precision'])}, Recall={len(performance_metrics['Recall'])}, F1-Score={len(performance_metrics['F1-Score'])}, MCC={len(performance_metrics['MCC'])}, Accuracy={len(performance_metrics['Accuracy'])}")

performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(performance_file_path, index=False)

# Get predictions for test set
test_predictions = svm_pipeline.predict(test_embeddings)

# Convert predictions back to label names
predicted_labels = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame with the results
results_df = pd.DataFrame({'id': [f'{i+1}' for i in range(len(test_data))], 'spoilerType': predicted_labels})

# Save the results to a CSV file
results_df.to_csv(output_file_path, index=False)
