import torch
import torchvision
from torchvision.models import ResNet34_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCR_Model(torch.nn.Module):
    def __init__(self, n_classes, out_len=16, weights=ResNet34_Weights.DEFAULT):
        super().__init__()

        # self.m = torchvision.models.resnet34(pretrained=True)
        # Инициализация модели ResNet34 с предварительно обученными весами
        self.m = torchvision.models.resnet34(weights=weights)
        
        self.blocks = [torch.nn.Conv2d(3, 64, 7, 1, 3), self.m.bn1, self.m.relu, self.m.maxpool,
                      self.m.layer1, self.m.layer2, self.m.layer3]
        self.feature_extractor = torch.nn.Sequential(*self.blocks)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((512, out_len))
        self.bilstm1 = torch.nn.LSTM(512, 256, 2, dropout=0.15, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, n_classes))

    def forward(self, x):
        feature = self.feature_extractor(x)
        b, c, h, w = feature.size()
        feature = feature.view(b, c * h, w)
        feature = self.avg_pool(feature)
        feature = feature.transpose(1, 2)
        out, (h_t1, c_t1) = self.bilstm1(feature)
        out = self.classifier(out)
        return out
