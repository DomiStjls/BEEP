from onnx_model import predict

from PIL import Image

inputsize = 128
sumiou = 0
cnt = 2765
for i in range(cnt):
    try:
        path1 = f"brain/train/images/{i}.jpg"
        mask = predict(path1)
        path2 = f"brain/train/masks/{i}.png"
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
                y_pred.append((m > 0.2))
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
        print(inter, union, iou)
    except:
        cnt -= 1
        print("error")
print(sumiou / cnt)
