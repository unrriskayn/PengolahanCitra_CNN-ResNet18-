# CATATAN
# JALANKAN KODE INI DULU, JIKA FILE (flower_classifier.pth) BELUM TERSEDIA
# Training CNN (ResNet18)

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Path dataset
data_dir = "D:\\ObjectDetections\\flower"  # Menyesuaikan letak penyimpanan
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

# Hitung jumlah gambar di setiap folder
def count_images_in_folder(folder):
    count = 0
    for subdir, _, files in os.walk(folder):
        count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count

print("Jumlah gambar:")
print(f"  Train     : {count_images_in_folder(train_dir)}")
print(f"  Validasi  : {count_images_in_folder(valid_dir)}")
print(f"  Test      : {count_images_in_folder(test_dir)}")

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset dan DataLoader
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# Model ResNet18
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Ganti FC Layer
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10):  # Ubah jumlah epoch sesuai kebutuhan
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Simpan model
torch.save(model.state_dict(), "flower_classifier.pth")


# ===== Evaluasi model: Akurasi & Confusion Matrix =====
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print("\nEvaluating model on validation data...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Hitung akurasi
acc = accuracy_score(all_labels, all_preds)
print(f"Akurasi Validasi: {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_data.classes, yticklabels=train_data.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.tight_layout()
plt.savefig("confusion_matrix_flower.png")
plt.show()
