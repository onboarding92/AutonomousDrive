import numpy as np
import cv2
import os

label_to_color = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

labelled = os.listdir("eval/ground_t/")
predictions = os.listdir("eval/pred/")


def get_mask(b, g, r, color, ran):
    cb = [color[0]]
    cg = [color[1]]
    cr = [color[2]]

    for i in range(ran):
        cb.append(color[0] - i)
        cb.append(color[0] + i)
        cg.append(color[1] - i)
        cg.append(color[1] + i)
        cr.append(color[2] - i)
        cr.append(color[2] + i)

    mask_b = b == 300
    mask_g = g == 300
    mask_r = r == 300
    for i in range(len(cb)):
        mask_b_p = b == cb[i]
        mask_b = np.logical_or(mask_b, mask_b_p)
        mask_g_p = g == cg[i]
        mask_g = np.logical_or(mask_g, mask_g_p)
        mask_r_p = r == cr[i]
        mask_r = np.logical_or(mask_r, mask_r_p)
    mask = np.logical_and(mask_b, mask_g)
    mask = np.logical_and(mask, mask_r)
    return mask


show = False
class_scores = [0] * len(label_to_color.keys())
class_count = [0] * len(label_to_color.keys())

for i in range(len(labelled)):
    print("READING IMAGE {}".format(i))
    img = cv2.imread("eval/ground_t/"+labelled[i])
    pred = cv2.imread("eval/pred/"+predictions[i])
    h, w = img.shape[:2]

    for cl, color in label_to_color.items():
        #print(color)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        mask = get_mask(b, g, r, color, 10)
        cl_mask = np.zeros_like(img)
        cl_mask[mask] = color
        if show:
            cv2.imshow("GROUNDTRUTH", cl_mask)
            cv2.waitKey(0)

        b_pred = pred[:, :, 0]
        g_pred = pred[:, :, 1]
        r_pred = pred[:, :, 2]
        mask_pred = get_mask(b_pred, g_pred, r_pred, color, 10)
        cl_mask_pred = np.zeros_like(img)
        cl_mask_pred[mask_pred] = color
        if show:
            cv2.imshow("PREDICTION", cl_mask_pred)
            cv2.waitKey(0)

        intersection = np.logical_and(cl_mask, cl_mask_pred)
        union = np.logical_or(cl_mask, cl_mask_pred)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
        if iou_score != 0:
            class_scores[cl] += iou_score
            class_count[cl] += 1

for cl in range(len(class_scores)):
    print("CLASS {} mIOU SCORE: {}".format(cl, class_scores[cl] / class_count[cl] if class_count[cl] != 0 else 0))
