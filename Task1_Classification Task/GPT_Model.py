import json
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
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

# Custom Dataset
class ClickbaitDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, label_encoder, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = label_encoder
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove special characters
        text = special_chars.sub(' ', text)
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return " ".join(filtered_tokens)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        postText = self.preprocess_text(entry.get('postText', [""])[0])
        targetParagraphs = self.preprocess_text(" ".join(entry.get('targetParagraphs', [])))
        targetTitle = self.preprocess_text(entry.get('targetTitle', ""))
        targetDescription = self.preprocess_text(entry.get('targetDescription', ""))
        targetKeywords = self.preprocess_text(entry.get('targetKeywords', ""))
        humanSpoiler = self.preprocess_text(entry.get('provenance', {}).get('humanSpoiler', ""))
        spoiler = self.preprocess_text(" ".join(entry.get('spoiler', [])))

        combined_text = " ".join([postText, targetParagraphs, targetTitle, targetDescription, targetKeywords, humanSpoiler, spoiler])
        
        inputs = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'id': self.data[idx].get('uuid', f'{idx}')  # Use uuid if exists, else assign a default value
        }
        
        if not self.is_test:
            label = entry['tags'][0]  # Use the first element of tags list as the label
            label = self.label_encoder.transform([label])[0]
            item['label'] = torch.tensor(label, dtype=torch.long)
        
        return item

# Load Data
def load_data(path, is_test=False):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if is_test or (isinstance(entry.get('tags'), list) and entry['tags'][0] in {'phrase', 'multi', 'passage'}):
                data.append(entry)
    return data

# Example paths (replace with actual file paths)
train_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\train.jsonl'
val_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\val.jsonl'
test_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\test.jsonl'
output_file_path = 'predictions_GPT.csv'  # File to save predictions
performance_file_path = 'performance_metrics_GPT.csv'  # File to save performance metrics
checkpoint_dir = r'C:\Users\chawl\Documents\ClickBait\Model_GPT_Task1\checkpoints'  # Directory to save checkpoints

# Create checkpoint directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_data = load_data(train_data_path)
val_data = load_data(val_data_path)
test_data = load_data(test_data_path, is_test=True)

# Tokenization and padding
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

max_len = 256

# Label encoding
label_encoder = LabelEncoder()
all_labels = [entry['tags'][0] for entry in train_data + val_data]
label_encoder.fit(all_labels)

train_dataset = ClickbaitDataset(train_data, tokenizer, max_len, label_encoder)
val_dataset = ClickbaitDataset(val_data, tokenizer, max_len, label_encoder)
test_dataset = ClickbaitDataset(test_data, tokenizer, max_len, label_encoder, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Model hyperparameters
OUTPUT_DIM = len(label_encoder.classes_)

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=OUTPUT_DIM)
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings for added pad token
model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model config knows about the pad token

# Training settings
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

# Load existing checkpoint if available
def load_checkpoint(model, checkpoint_dir, tokenizer):
    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pth')):
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
        model.resize_token_embeddings(len(tokenizer))  # Ensure token embeddings match
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Reinitialize optimizer and manually adjust its state
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Adjust optimizer state to match the new embedding size
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.shape[0] == checkpoint['model_state_dict']['transformer.wte.weight'].shape[0]:
                        new_size = model.transformer.wte.weight.shape[0]
                        if v.shape[0] != new_size:
                            state[k] = torch.cat((v, v.new_zeros(new_size - v.shape[0], *v.shape[1:])))
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        optimizer = optim.Adam(model.parameters())  # Initialize optimizer if no checkpoint
        start_epoch = 0
    return model, optimizer, start_epoch

model, optimizer, start_epoch = load_checkpoint(model, checkpoint_dir, tokenizer)

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pth'))
    print(f"Checkpoint saved at epoch {epoch}")

def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

N_EPOCHS = 3

for epoch in range(start_epoch, N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}')
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)

def get_predictions(model, loader, is_test=False):
    model.eval()
    predictions = []
    labels = []
    ids = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            ids.extend(batch['id'])
            if not is_test:
                labels.extend(batch['label'].cpu().numpy())
            
    if is_test:
        return predictions, ids
    else:
        return predictions, labels, ids

# Get predictions for validation set
val_predictions, val_labels, val_ids = get_predictions(model, val_loader)

# Ensure labels and predictions have the same length
val_labels = val_labels[:len(val_predictions)]

# Evaluate the model
class_labels = label_encoder.classes_

performance_metrics = {
    'Class': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'MCC': [],
    'Accuracy': []
}

# Compute class-specific metrics
for i, label in enumerate(class_labels):
    precision = precision_score(val_labels, val_predictions, labels=[i], average='weighted', zero_division=0)
    recall = recall_score(val_labels, val_predictions, labels=[i], average='weighted', zero_division=0)
    f1 = f1_score(val_labels, val_predictions, labels=[i], average='weighted', zero_division=0)
    mcc = matthews_corrcoef(val_labels, val_predictions) if len(set(val_predictions)) > 1 else 0
    accuracy = accuracy_score(val_labels, [i if p == i else -1 for p in val_predictions])

    performance_metrics['Class'].append(label)
    performance_metrics['Precision'].append(precision)
    performance_metrics['Recall'].append(recall)
    performance_metrics['F1 Score'].append(f1)
    performance_metrics['MCC'].append(mcc)
    performance_metrics['Accuracy'].append(accuracy)

# Compute weighted metrics
weighted_precision = precision_score(val_labels, val_predictions, average='weighted')
weighted_recall = recall_score(val_labels, val_predictions, average='weighted')
weighted_f1 = f1_score(val_labels, val_predictions, average='weighted')
weighted_mcc = matthews_corrcoef(val_labels, val_predictions) if len(set(val_predictions)) > 1 else 0
weighted_accuracy = accuracy_score(val_labels, val_predictions)

# Append weighted metrics to performance_metrics dictionary
performance_metrics['Class'].append('Weighted')
performance_metrics['Precision'].append(weighted_precision)
performance_metrics['Recall'].append(weighted_recall)
performance_metrics['F1 Score'].append(weighted_f1)
performance_metrics['MCC'].append(weighted_mcc)
performance_metrics['Accuracy'].append(weighted_accuracy)

# Debugging: Print lengths of all lists
for key, value in performance_metrics.items():
    print(f"{key}: {len(value)}")

performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(performance_file_path, index=False)

print(performance_df)

# Get predictions for test set
test_predictions, test_ids = get_predictions(model, test_loader, is_test=True)

# Convert predictions back to label names
predicted_labels = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame with the results
results_df = pd.DataFrame({'id': [f'{i}' for i in range(len(test_ids))], 'spoilerType': predicted_labels})

# Save the results to a CSV file
results_df.to_csv(output_file_path, index=False)
