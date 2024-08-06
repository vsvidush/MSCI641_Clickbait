import json
import re
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
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

class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Load Data
train_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\train.jsonl'
val_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\val.jsonl'
test_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\test.jsonl'
output_file_path = 'predictions_SVM_RoBERTa.csv'  # File to save predictions
performance_file_path = 'performance_metrics_SVM_RoBERTa.csv'  # File to save performance metrics
checkpoint_dir = r'C:\Users\chawl\Documents\ClickBait\Model_RobertSVM_Task1\checkpoints'  # Directory to save checkpoints

os.makedirs(checkpoint_dir, exist_ok=True)

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

# Load RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuning settings
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Create datasets and dataloaders
train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = ClickbaitDataset(test_texts, None, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define optimizer and scheduler
optimizer = AdamW(roberta_model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Function to save checkpoints
def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

# Function to load checkpoints
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
    if not checkpoints:
        return 0
    checkpoints.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Loaded checkpoint: {checkpoint_path}')
    return epoch + 1

# Load checkpoint if exists
start_epoch = load_checkpoint(roberta_model, optimizer, scheduler, checkpoint_dir)
roberta_model = roberta_model.to(device)

# Fine-tuning function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = loss_fn(outputs.logits, labels)
        correct_predictions += torch.sum(torch.argmax(outputs.logits, dim=1) == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs.logits, labels)
            correct_predictions += torch.sum(torch.argmax(outputs.logits, dim=1) == labels)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

for epoch in range(start_epoch, EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS + start_epoch}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        roberta_model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_dataset)
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(
        roberta_model,
        val_loader,
        loss_fn,
        device,
        len(val_dataset)
    )
    
    print(f'Val   loss {val_loss} accuracy {val_acc}')

    # Save checkpoint
    save_checkpoint(roberta_model, optimizer, scheduler, epoch + start_epoch, checkpoint_dir)

# Generate RoBERTa embeddings using the fine-tuned model
def encode_texts(texts, model, tokenizer):
    embeddings = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LEN).to(device)
            outputs = model.roberta(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

train_embeddings = encode_texts(train_texts, roberta_model, tokenizer)
val_embeddings = encode_texts(val_texts, roberta_model, tokenizer)
test_embeddings = encode_texts(test_texts, roberta_model, tokenizer)

# Combine training and validation data for SVM training
X = np.vstack([train_embeddings, val_embeddings])
y = np.hstack([train_labels, val_labels])

# Create SVM model pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC(kernel='linear', C=1.0, random_state=42))
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

# Create a DataFrame with the results, adding ID in sequence
results_df = pd.DataFrame({'id': [f'{i+1}' for i in range(len(predicted_labels))], 'spoilerType': predicted_labels})

# Save the results to a CSV file
results_df.to_csv(output_file_path, index=False)
