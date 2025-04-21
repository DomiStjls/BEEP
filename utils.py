import torch
import torchvision.transforms as transforms
from PIL import Image

# Загрузка модели
model = torch.load("model/model.pth", map_location=torch.device("cpu"))
model.eval()

# Преобразования для входного изображения
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(),  # если МРТ в градациях серого
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def predict_tumor(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # добавляем batch dim
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    return "Опухоль обнаружена" if prediction > 0.5 else "Опухоль не обнаружена"
