import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F

# ⚙️ Настройки
MODEL_PATH = "model/best_model.pth"
INPUT_SIZE = (128, 128)  # Размер, с которым обучалась модель

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🧠 Тот же MiniUNet, что и при обучении
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super(MiniUNet, self).__init__()
        self.enc1 = CBR(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(base_filters * 2, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = CBR(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = CBR(base_filters * 2, base_filters)
        self.final = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.final(d1))

def unit_mini(image_path):
    # 📥 Загрузка модели
    model = MiniUNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 📸 Препроцессинг изображения
    transform = transforms.Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor()])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # 🔍 Предсказание
    with torch.no_grad():
        output = model(input_tensor)
        mask_pred = output.squeeze().cpu().numpy()

    original_img = Image.open(image_path).convert("RGB").resize((128, 128))
    return mask_pred, original_img
# 🖼️ Отображение результата
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(img)

# plt.subplot(1, 2, 2)
# plt.title("Predicted Mask")
# plt.imshow(mask_pred, cmap="gray")

# plt.tight_layout()
# plt.show()
