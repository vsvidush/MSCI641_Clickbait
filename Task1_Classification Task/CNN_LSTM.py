import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        
        inputs = self.tokenizer.encode_plus(
            combined_text,
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
            'id': self.data[idx].get('uuid', f'missing_{idx}')  # Use uuid if exists, else assign a default value
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

# Model Definition
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, num_filters, filter_sizes):
        super(CNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes])
        self.lstm = nn.LSTM(num_filters * len(filter_sizes), hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        embedded = embedded.unsqueeze(1)  # Add channel dimension for CNN
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = cat.unsqueeze(1)  # Add back channel dimension for LSTM
        
        # Calculate the proper sequence length after the CNN layers
        seq_len = cat.size(1)
        text_lengths = torch.full((text_lengths.size(0),), seq_len, dtype=torch.long)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(cat, text_lengths.cpu(), enforce_sorted=False, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        return self.fc(hidden)

# Example paths (replace with actual file paths)
train_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\train.jsonl'
val_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\val.jsonl'
test_data_path = r'C:\Users\chawl\Documents\ClickBait\task-1-clickbait-detection-msci-641-s-24\test.jsonl'
output_file_path = 'predictions_CNN_LSTM.csv'  # File to save predictions
performance_file_path = 'performance_metrics_CNN_LSTM.csv'  # File to save performance metrics

train_data = load_data(train_data_path)
val_data = load_data(val_data_path)
test_data = load_data(test_data_path, is_test=True)

# Tokenization and padding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 256

# Label encoding
label_encoder = LabelEncoder()
all_labels = [entry['tags'][0] for entry in train_data + val_data]
label_encoder.fit(all_labels)

train_dataset = ClickbaitDataset(train_data, tokenizer, max_len, label_encoder)
val_dataset = ClickbaitDataset(val_data, tokenizer, max_len, label_encoder)
test_dataset = ClickbaitDataset(test_data, tokenizer, max_len, label_encoder, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model hyperparameters
INPUT_DIM = len(tokenizer.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(label_encoder.classes_)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]

model = CNN_LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, NUM_FILTERS, FILTER_SIZES)

# Training settings
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

# Early stopping
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        text = batch['input_ids'].to(device)
        text_lengths = batch['attention_mask'].sum(dim=1).to(device)  # Calculate sequence lengths
        model.lstm.flatten_parameters()  # Ensure LSTM weights are contiguous
        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            text = batch['input_ids'].to(device)
            text_lengths = batch['attention_mask'].sum(dim=1).to(device)  # Calculate sequence lengths
            model.lstm.flatten_parameters()  # Ensure LSTM weights are contiguous
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch['label'].to(device))
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)
N_EPOCHS = 15
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

def get_predictions(model, loader, is_test=False):
    model.eval()
    predictions = []
    labels = []
    ids = []
    
    with torch.no_grad():
        for batch in loader:
            text = batch['input_ids'].to(device)
            text_lengths = batch['attention_mask'].sum(dim=1).to(device)  # Calculate sequence lengths
            preds = model(text, text_lengths)
            predictions.extend(torch.argmax(preds, dim=1).cpu().numpy())
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
accuracy = accuracy_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions, average='weighted')
recall = recall_score(val_labels, val_predictions, average='weighted')
f1 = f1_score(val_labels, val_predictions, average='weighted')

print(f'Validation Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')

# Save performance metrics to CSV
performance_metrics = {
    'Model': ['CNN_LSTM'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
}

# Use a different filename for the performance metrics to avoid permission issues
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('performance_metrics_CNN_LSTM.csv', index=False)

# Get predictions for test set
test_predictions, test_ids = get_predictions(model, test_loader, is_test=True)

# Convert predictions back to label names
predicted_labels = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame with the results
results_df = pd.DataFrame({'id': test_ids, 'spoilerType': predicted_labels})

# Save the results to a CSV file
results_df.to_csv(output_file_path, index=False)
