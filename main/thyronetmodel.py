import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224
class_names = ['TIRADS 1', 'TIRADS 2', 'TIRADS 3', 'TIRADS 4', 'TIRADS 5']

root_dir = os.path.dirname(os.path.abspath(__file__))

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ThyroidClassifierV6(nn.Module):
    def __init__(self, num_classes=5):
        super(ThyroidClassifierV6, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.transformer_dim = 512 * Bottleneck.expansion
        self.num_tokens = 7 * 7
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=8,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(self.transformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)

        return x
    
def load_model_1(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model {checkpoint_path} loaded successfully !!")
    model.to(device)
    model.eval()
    return model

def load_model_2(file_path=os.path.join(root_dir,"xgb_model.pkl")):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Model {file_path} loaded successfully !!")
    return data["model"], data["scaler"], data["label_encoder"]

tmodel = ThyroidClassifierV6(num_classes=5)
tmodel = load_model_1(tmodel, os.path.join(root_dir,"ThyroidClassifierV6.pth"))

loaded_model, loaded_scaler, loaded_label_encoder = load_model_2()

def predict_tirads(image, model = tmodel, transform = val_transform):
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = {f"TIRADS{i+1}" : round(float(torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()[i]*100), 2) for i in range(0,5)}
        _, predicted = torch.max(outputs, 1)
    return [class_names[predicted.item()], probabilities]

def predict_condition(new_input):

    loaded_model, loaded_scaler, loaded_label_encoder = load_model_2()

    input_df = pd.DataFrame([new_input])
    input_df['Sex'] = input_df['Sex'].map({'Female': 0, 'Male': 1})
    input_scaled = loaded_scaler.transform(input_df)
    
    probabilities = loaded_model.predict_proba(input_scaled)[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = loaded_label_encoder.inverse_transform([predicted_index])[0]
    
    class_probabilities = {loaded_label_encoder.inverse_transform([i])[0]: round(float(prob)*100, 2) for i, prob in enumerate(probabilities)}
    
    if class_probabilities[predicted_class] < 30:
        predicted_class = None

    return [predicted_class, class_probabilities]