import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# Load data 
X = pd.read_csv('data/processed/X_genomic.csv')
y = pd.read_csv('data/processed/y.csv')
y = y.drop(columns=['sampleId', 'patientId']).astype(int)

X = X.drop(columns=['sampleId'])
X = X.values.astype(np.float32)
y = y.values.astype(np.float32)

LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']
N_LABELS = 7
INPUT_DIM = X.shape[1]  # 69

print(f'X shape: {X.shape}, y shape: {y.shape}')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Train size: {len(X_train)} | Test size: {len(X_test)}')
print(f'Positives per label (train): {y_train.sum(axis=0)}')
print(f'Positives per label (test):  {y_test.sum(axis=0)}')

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Pos weights for class imbalance
def compute_pos_weights(y_train):
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)

# MLP architecture 
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, n_labels=7):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden2, n_labels)
        )

    def forward(self, x):
        return self.network(x)
    
# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Build dataloader 
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
EPOCHS = 200
LEARNING_RATE = 1e-3

pos_weights = compute_pos_weights(y_train)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

model = MLP(input_dim=INPUT_DIM)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-2
)

print('\nTraining MLP...')
for epoch in range(EPOCHS):
    loss = train_epoch(model, loader, optimizer, criterion)
    if (epoch + 1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}')

# Evaluate 
def predict(model, X_scaled):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
        return (probs > 0.5).astype(int)

y_pred_binary = predict(model, X_test_scaled)
y_train_pred_binary = predict(model, X_train_scaled)

test_f1 = f1_score(y_test.astype(int), y_pred_binary, average='macro', zero_division=0)
train_f1 = f1_score(y_train.astype(int), y_train_pred_binary, average='macro', zero_division=0)
per_label_f1 = f1_score(y_test.astype(int), y_pred_binary, average=None, zero_division=0)
test_acc = accuracy_score(y_test.astype(int), y_pred_binary)
train_acc = accuracy_score(y_train.astype(int), y_train_pred_binary)


# Print results 
print('\n' + '='*60)
print('MLP RESULTS')
print('='*60)
print(f'Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {train_f1 - test_f1:+.3f}')
print(f'Test Accuracy: {test_acc:.3f} | Train Accuracy: {train_acc:.3f}')
print('\nPer-label F1:')


mlp_results = {
    'test_f1': test_f1,
    'test_accuracy': test_acc,
    'train_f1': train_f1,
    'train_accuracy': train_acc,
}

for label, score in zip(LABEL_NAMES, per_label_f1):
    print(f'  {label:<10} {score:.3f}')
    mlp_results[f'test_f1_{label.lower()}'] = score

# Save results
os.makedirs('results', exist_ok=True)

mlp_df = pd.DataFrame({'MLP': mlp_results})
mlp_df.to_csv('results/evaluation_genomic_mlp.csv')
print('\nSaved: results/evaluation_genomic_mlp.csv')


