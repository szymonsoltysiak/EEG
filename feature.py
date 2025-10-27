import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data_prep import split_csv_train_test

# --- Dataset class ---
class EEGDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        # Encode string labels to integers
        self.label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.labels = [self.label_map[lbl] for lbl in labels]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return torch.tensor(self.frames[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# --- Feature extraction ---
def extract_psd_features(batch_data, fs=128, nperseg=64):
    batch_data = batch_data.cpu().numpy() if isinstance(batch_data, torch.Tensor) else batch_data
    psd_features = []
    for x in batch_data:
        channel_psds = []
        for ch in range(x.shape[1]):
            f, Pxx = welch(x[:, ch], fs=fs, nperseg=nperseg)
            channel_psds.append(Pxx)
        psd_features.append(np.stack(channel_psds, axis=0))  # (channels, freqs)
    return torch.tensor(np.stack(psd_features), dtype=torch.float32)

# --- CNN Model ---
class EEGCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- Load CSVs and create train/val/test splits ---
def load_eeg_data(csv_folder, window_size=128, coverage=64, test_ratio=0.2, val_ratio=0.1):
    train_frames, train_labels = [], []
    val_frames, val_labels = [], []
    test_frames, test_labels = [], []

    csv_filepaths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

    for csv_file in csv_filepaths:
        df = pd.read_csv(csv_file)

        # Split raw df
        train_val_df, test_df = split_csv_train_test(df, test_ratio=test_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42, stratify=train_val_df['Annotation'])

        # Create sliding windows
        def create_frames(df):
            frames, labels = [], []
            data = df.drop(columns=['Annotation']).values
            label_vals = df['Annotation'].values
            step = window_size - coverage
            for start in range(0, len(df) - window_size + 1, step):
                frames.append(data[start:start+window_size])
                labels.append(label_vals[start + window_size // 2])
            return frames, labels

        t_frames, t_labels = create_frames(train_df)
        v_frames, v_labels = create_frames(val_df)
        te_frames, te_labels = create_frames(test_df)

        # Append to global lists
        train_frames.extend(t_frames)
        train_labels.extend(t_labels)
        val_frames.extend(v_frames)
        val_labels.extend(v_labels)
        test_frames.extend(te_frames)
        test_labels.extend(te_labels)

    print("Train class distribution:", dict(Counter(train_labels)))
    print("Validation class distribution:", dict(Counter(val_labels)))
    print("Test class distribution:", dict(Counter(test_labels)))

    return EEGDataset(train_frames, train_labels), EEGDataset(val_frames, val_labels), EEGDataset(test_frames, test_labels)

# --- Train/test setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset, val_dataset, test_dataset = load_eeg_data('datasets/mind_drive/csv_files')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Model setup ---
dummy_batch, _ = next(iter(train_loader))
dummy_psd = extract_psd_features(dummy_batch)
n_channels, n_freqs = dummy_psd.shape[1], dummy_psd.shape[2]
n_classes = len(set(train_dataset.labels))

model = EEGCNN(n_channels, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# --- Training loop ---
for epoch in range(20):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        data = extract_psd_features(data).to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    val_correct, val_total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            data = extract_psd_features(data).to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total * 100
    print(f"Epoch {epoch+1}, Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# --- Test evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        data = extract_psd_features(data).to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")
