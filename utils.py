
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


def get_polygon(image_path):
    return [0.132352940625, 0.66135458125, 0.1813725484375, 0.749003984375, 0.24019607812500002, 0.828685259375, 0.3529411765625, 0.8725099609375, 0.4607843140625, 0.8645418328125001, 0.5245098046875001, 0.8247011953125, 0.563725490625, 0.74103585625, 0.6176470593750001, 0.6653386453125, 0.6176470593750001, 0.5936254984375, 0.5686274515624999, 0.490039840625, 0.5, 0.410358565625, 0.348039215625, 0.3745019921875, 0.1764705875, 0.509960159375, 0.12254902031249999, 0.5697211156249999, 0.132352940625, 0.66135458125]

def predict_tumor(image_path):
    image = prepare_image_tf(image_path)
    prediction = model.predict(image)[0][0]
    return "Tumor detected 🚨" if prediction > 0.5 else "No tumor detected ✅"
