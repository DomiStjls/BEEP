
from PIL import Image
from keras.models import load_model
import numpy as np

model = load_model("model/brain_mri_cnn_model1.h5")


def prepare_image_tf(image_path):
    img = Image.open(image_path).convert("L")  # 'L' for grayscale
    img = img.resize((128, 128))  # обязательный resize
    img_array = np.array(img).astype("float32") / 255.0  # нормализация
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 1)
    return img_array

def predict_tumor(image_path):
    image = prepare_image_tf(image_path)
    prediction = model.predict(image)[0][0]
    return "Tumor detected 🚨" if prediction > 0.5 else "No tumor detected ✅"
