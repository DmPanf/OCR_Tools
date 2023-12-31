import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Предположим, что у нас есть папка 'data' с подпапками для каждого класса
# Предобработка и загрузка данных
transforms = Compose([Resize((256, 256)), ToTensor()])
dataset = ImageFolder(root='data', transform=transforms)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Создание модели
n_classes = 10 # для цифр от 0 до 9
model = OCR_Model(n_classes).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 10
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Валидация модели
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}")
