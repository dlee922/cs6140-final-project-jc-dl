import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Load data
X = pd.read_csv('data/processed/X_combined.csv')
y = pd.read_csv('data/processed/y.csv')
y = y.drop(columns=['sampleId', 'patientId']).astype(int)

X = X.drop(columns=['patientId', 'sampleId', 'GENE_PANEL'])
X = X.values.astype(np.float32)
y = y.values.astype(np.float32)

LABEL_NAMES = ['Adrenal', 'Bone', 'CNS', 'Liver', 'LN', 'Lung', 'Pleura']
CHAIN_ORDER = [2, 1, 3, 0, 5, 6, 4]
N_LABELS = 7
INPUT_DIM = X.shape[1]
AUGMENTED_DIM = INPUT_DIM + N_LABELS

print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'Augmented input dim: {AUGMENTED_DIM}')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Train size: {len(X_train)} | Test size: {len(X_test)}')
print(f'Positives per label (train): {y_train.sum(axis=0)}')
print(f'Positives per label (test):  {y_test.sum(axis=0)}')

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pos weights for class imbalance
def compute_pos_weights(y_train):
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)

# MLP architecture
class ChainMLP(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, n_labels=7):
        super(ChainMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, n_labels)
        )

    def forward(self, x):
        return self.network(x)

# Training function
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch, _ in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Augmented input builder 
def build_augmented_input(X, y_partial):
    return np.concatenate([X, y_partial], axis=1).astype(np.float32)

# Build augmented training inputs via teacher forcing 
pos_weights = compute_pos_weights(y_train)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

augmented_X_train_list = []
augmented_y_train_list = []

for k in range(N_LABELS):
    y_partial = np.zeros_like(y_train)
    for prev in range(k):
        label_idx = CHAIN_ORDER[prev]
        y_partial[:, label_idx] = y_train[:, label_idx]
    aug_X = build_augmented_input(X_train_scaled, y_partial)
    augmented_X_train_list.append(aug_X)
    augmented_y_train_list.append(y_train)

X_aug = np.vstack(augmented_X_train_list)
y_aug = np.vstack(augmented_y_train_list)

X_aug_tensor = torch.tensor(X_aug, dtype=torch.float32)
y_aug_tensor = torch.tensor(y_aug, dtype=torch.float32)
dummy_mask = torch.zeros(len(X_aug))
dataset = TensorDataset(X_aug_tensor, y_aug_tensor, dummy_mask)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train 
EPOCHS = 200
LEARNING_RATE = 1e-3

model = ChainMLP(input_dim=AUGMENTED_DIM)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

print('\nTraining MLP...')
for epoch in range(EPOCHS):
    loss = train_model(model, loader, optimizer, criterion)
    if (epoch + 1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}')

# Evaluate on test set 
model.eval()
with torch.no_grad():
    y_pred_binary = np.zeros_like(y_test, dtype=int)

    for k in range(N_LABELS):
        y_partial = np.zeros_like(y_test, dtype=np.float32)
        for prev in range(k):
            label_idx = CHAIN_ORDER[prev]
            y_partial[:, label_idx] = y_pred_binary[:, label_idx]

        aug_X_test = build_augmented_input(X_test_scaled, y_partial)
        X_test_tensor = torch.tensor(aug_X_test, dtype=torch.float32)

        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy()

        target_idx = CHAIN_ORDER[k]
        y_pred_binary[:, target_idx] = (probs[:, target_idx] > 0.5).astype(int)

# Evaluate on train set (overfitting check)
model.eval()
with torch.no_grad():
    y_train_pred_binary = np.zeros((len(y_train), N_LABELS), dtype=int)

    for k in range(N_LABELS):
        y_partial = np.zeros((len(y_train), N_LABELS), dtype=np.float32)
        for prev in range(k):
            label_idx = CHAIN_ORDER[prev]
            y_partial[:, label_idx] = y_train_pred_binary[:, label_idx]

        aug_X_train = build_augmented_input(X_train_scaled, y_partial)
        X_train_tensor = torch.tensor(aug_X_train, dtype=torch.float32)

        logits = model(X_train_tensor)
        probs = torch.sigmoid(logits).numpy()

        target_idx = CHAIN_ORDER[k]
        y_train_pred_binary[:, target_idx] = (probs[:, target_idx] > 0.5).astype(int)

# Results 
test_f1 = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
train_f1 = f1_score(y_train, y_train_pred_binary, average='macro', zero_division=0)
per_label_f1 = f1_score(y_test, y_pred_binary, average=None, zero_division=0)
test_acc = accuracy_score(y_test, y_pred_binary)

print('\n' + '='*60)
print('MLP RESULTS')
print('='*60)
print(f'Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {train_f1 - test_f1:+.3f}')
print(f'Test Accuracy: {test_acc:.3f}')
print('\nPer-label F1:')
for label, score in zip(LABEL_NAMES, per_label_f1):
    print(f'  {label:<10} {score:.3f}')
