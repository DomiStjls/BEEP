
from PIL import Image
from keras.models import load_model  # type: ignore
import numpy as np
from unet_mini import unit_mini

model = load_model("model/brain_mri_cnn_model1.h5")


def prepare_image_tf(image_path):
    img = Image.open(image_path).convert("L")  # 'L' for grayscale
    img = img.resize((128, 128))  # обязательный resize
    img_array = np.array(img).astype("float32") / 255.0  # нормализация
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 1)
    return img_array

def get_img(image_path):
    mask, orig = unit_mini(image_path)
    size = len(mask)
    pixels = orig.load()
    print(mask.min(), mask.max())
    for i in range(size):
        for j in range(size):
            pixels[j, i] = ((255 - int(mask[i][j] * 255)) * 20, 0, 0)
    orig.save(image_path)

def predict_tumor(image_path):
    image = prepare_image_tf(image_path)
    prediction = model.predict(image)[0][0]
    if prediction <= 0.5:
        return "No tumor detected ✅"
    get_img(image_path)
    return "Tumor detected 🚨"

