
from PIL import Image
from keras.models import load_model  # type: ignore
import numpy as np
from onnx_model import predict

model = load_model("model/brain_mri_cnn_model1.h5")

INPUT_SIZE = (128, 128)

def prepare_image_tf(image_path):
    img = Image.open(image_path).convert("L")  # 'L' for grayscale
    img = img.resize((128, 128))  # обязательный resize
    img_array = np.array(img).astype("float32") / 255.0  # нормализация
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 1)
    return img_array

def get_img(image_path):
    mask = predict(image_path)
    size = len(mask)
    orig = Image.open(image_path).convert("RGB").resize(INPUT_SIZE)
    pixels = orig.load()
    minim = mask.min()
    maxim = mask.max()
    print(minim, maxim)
    for i in range(size):
        for j in range(size):
            val = (255, 0, 0)
            m = 1 - (mask[j][i] - minim) / (maxim - minim)
            cur = pixels[i, j]
            pixels[i, j] = (int(cur[0] - (cur[0] - val[0]) * m), int(cur[1] - (cur[1] - val[1]) * m), int(cur[2] - (cur[2] - val[2]) * m))
    orig.save(image_path[:-4] + '1.jpg')

def predict_tumor(image_path):
    image = prepare_image_tf(image_path)
    prediction = model.predict(image)[0][0]
    if prediction <= 0.5:
        return "Низкая вероятность наличия опухоли ✅"
    get_img(image_path)
    return "Высокая вероятность наличия опухоли 🚨"

