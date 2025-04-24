import onnxruntime as ort
import numpy as np
from PIL import Image

MODEL_PATH = "model/best_model3.onnx"
value = 128
canal = 1



INPUT_SIZE = (value, value)


class ONNXTumorSegmenter:
    def __init__(self, model_path: str, img_size: int = value):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Преобразование PIL Image в тензор ONNX (1, 3, H, W)"""
        image = image.convert("L").resize((self.img_size, self.img_size))
        np_img = np.array(image).astype(np.float32) / 255.0
        np_img = np.stack([np_img] * canal, axis=0)  # (3, H, W)
        np_img = np.expand_dims(np_img, axis=0)  # (1, 3, H, W)
        return np_img

    def predict(self, image: Image.Image) -> np.ndarray:
        """Возвращает маску (массив 2D)"""
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        mask = outputs[0][0, 0]  # (H, W)
        return mask


def predict(image_path):
    image = Image.open(image_path).convert("L" if canal == 1 else "RGB").resize(INPUT_SIZE)
    segmenter = ONNXTumorSegmenter(MODEL_PATH)
    mask = segmenter.predict(image)
    return mask


# print(predict("static/bobritobandito.png"))