from data_prep import df_to_dataset_random_chunk
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# --- Hyperparameters ---
from EEGTransformer import EEGClassifier, params # choose your model here from EEGTransformer/EEGNet/EEGBase
coverage = 64  # overlap in frames


batch_size = params['batch_size']
learning_rate = params['learning_rate']
num_epochs = params['num_epochs']
num_classes = params['num_classes']
window_size = params['window_size']
num_channels = params['num_channels']
step_size = params['step_size']
gamma = params['gamma']

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load data ---
data_path = 'datasets/mind_drive/csv_files'
csv_filepaths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
train_dataset, val_dataset, test_dataset = df_to_dataset_random_chunk(csv_filepaths, windowSize=128, coverage=coverage, test_ratio=0.2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Model, loss, optimizer ---
model = EEGClassifier(num_channels=num_channels, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- Training loop ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0  # counters for accuracy

    for data, labels in train_loader:
        data = data.to(device)           # (batch, timesteps, channels)
        labels = labels.to(device)

        data = data.permute(0, 2, 1)    # (batch, channels, timesteps)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

        # --- calculate training accuracy ---
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total * 100

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            data = data.permute(0, 2, 1)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    scheduler.step()


# --- Evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        data = data.permute(0, 2, 1)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")
