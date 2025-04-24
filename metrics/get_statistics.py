from onnx_model_copy import predict, size
from PIL import Image

PATH = "test"
cnt = 100


inputsize = size
sumiou = 0
fail = 0
for i in range(cnt):
    try:
        path1 = f"brain/{PATH}/images/{i}.jpg"
        mask = predict(path1)
        path2 = f"brain/{PATH}/masks/{i}.png"
        img2 = Image.open(path2).resize((inputsize, inputsize))
        img2 = img2.load()
        val = 0
        minim = mask.min()
        maxim = mask.max()
        y_pred = []
        y_true = []
        for x in range(inputsize):
            for y in range(inputsize):
                m = 1 - (mask[y][x] - minim) / (maxim - minim)
                y_pred.append((m > 0.5))
                y_true.append((img2[x, y][1] < 200))
        # print(y_true)
        # print(y_pred)
        inter = sum([(y_pred[i] and y_true[i]) for i in range(len(y_pred))])
        union = sum([(y_pred[i] or y_true[i]) for i in range(len(y_pred))])
        if union == 0:
            iou = (1.0 if inter == 0 else 0.0)
            sumiou += iou
        else:
            iou = inter / union
            sumiou += iou
        print(i, inter, union, iou, sumiou / (i + 1 - fail))
    except Exception as e:
        fail += 1
        if 'Keyboard' in str(e):
            break
        print(e)
