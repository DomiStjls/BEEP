from onnx_model_copy import predict, size
from PIL import Image

PATH = "test"
cnt = 123


inputsize = size
sumiou = 0
sumf1 = 0
summap50 = 0
cntmap50 = 0
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
        else:
            iou = inter / union
        sumiou += iou
        if iou > 0.5:
            summap50 += iou
            cntmap50 += 1
        tp = sum([(y_pred[i] and y_true[i]) for i in range(len(y_pred))])
        tn = sum([(not y_pred[i] and not y_true[i]) for i in range(len(y_pred))])
        fn = sum([(not y_pred[i] and y_true[i]) for i in range(len(y_pred))])
        fp = sum([(y_pred[i] and not y_true[i]) for i in range(len(y_pred))])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1score = 0
        else:
            f1score = 2 * precision * recall / (precision + recall)
        sumf1 += f1score
        n = i + 1 - fail
        print(f"{i}: {f1score:.5f} {iou:.5f}")
        # print(f"f1 = {(sumf1 / n):.5f}, iou = {(sumiou / n):.5f}, map50 = {(summap50 / cntmap50):.5f}")
    except Exception as e:
        fail += 1
        if 'Keyboard' in str(e):
            break
        print(e)

n = cnt - fail
print("Final:")
print(f"F1 score: {sumf1 / n}, IOU: {sumiou / n}, mAP50: {(summap50 / cntmap50) if cntmap50 != 0 else -1}")