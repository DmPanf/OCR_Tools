import torch
import torch.nn as nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=128, rnn_num_layers=2):
        super(CRNN, self).__init__()
        
        # Инициализация базовой CNN (например, ResNet)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # Удаление полносвязного слоя и слоя усреднения
        self.cnn = nn.Sequential(*modules)
        
        # Инициализация RNN
        self.rnn = nn.LSTM(
            input_size=512,  # Зависит от выхода CNN
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True
        )
        
        # Классификатор на выходе RNN
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)  # Умножаем на 2 из-за двунаправленности RNN

    def forward(self, x):
        # Извлечение признаков CNN
        features = self.cnn(x)
        
        # Подготовка признаков для RNN
        features = features.view(features.size(0), features.size(1), -1)  # Изменение формы для RNN
        features = features.permute(2, 0, 1)  # Перестановка для соответствия входу RNN
        
        # Пропускаем через RNN
        rnn_out, _ = self.rnn(features)
        
        # Преобразование выхода RNN для классификации
        rnn_out = rnn_out.view(rnn_out.size(1), rnn_out.size(2))  # Снова изменение формы для классификатора
        output = self.classifier(rnn_out)
        
        return output

# Пример использования
num_classes = 10  # Примерное количество классов для распознавания
crnn = CRNN(num_classes)

# Допустим, у нас есть загруженные данные в переменной data_loader
optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Процесс обучения
for images, labels in data_loader:
    optimizer.zero_grad()
    outputs = crnn(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
