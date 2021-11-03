import os

import cv2
import numpy as np
from tqdm import tqdm


def xywh2xyxy(xywh):
    xyxy = [int(xywh[0]),
            int(xywh[1]),
            int(xywh[0] + xywh[2]),
            int(xywh[1] + xywh[3])]
    return xyxy


def draw_xyxy(xyxy, img, color=None, thickness=2):
    color = np.random.choice(range(256), size=3).tolist() if color is None else color
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color, thickness=thickness)


def draw_xywh(cpwh, img, color=None, thickness=2):
    xyxy = xywh2xyxy(cpwh)
    draw_xyxy(xyxy, img, color, thickness)


def get_xyxy_e(box, w, h):
    xyxy = xywh2xyxy(box)
    xyxy_e = [max(0, xyxy[0]),
              max(0, xyxy[1]),
              min(w, xyxy[2]),
              min(h, xyxy[3])]
    return xyxy_e


def check_xyxy(xyxy):
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    if w > 0 and h > 0:
        return True
    else:
        return False

def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round(((xyxy[0] + xyxy[2]) / 2 / w), 6),
             round(((xyxy[1] + xyxy[3]) / 2 / h), 6),
             round(((xyxy[2] - xyxy[0]) / w), 6),
             round(((xyxy[3] - xyxy[1]) / h), 6)]
    return cpwhn


if __name__ == "__main__":
    img_root = "/home/daton/Downloads/leftImg8bit_trainvaltest/leftImg8bit"
    annot_root = "/home/daton/Downloads/gtBbox_cityPersons_trainval/gtBboxCityPersons"
    target_dir = "train"
    target_cities = sorted(os.listdir(os.path.join(annot_root, target_dir)))

    out_root = "/home/daton/Downloads/citypersons"
    for city in target_cities:
        imgs = sorted(os.listdir(os.path.join(img_root, target_dir, city)))
        annots = sorted(os.listdir(os.path.join(annot_root, target_dir, city)))
        print(len(imgs), len(annots))
        print(imgs)
        print(annots)

        for img_file, annot_file in zip(imgs, annots):
            img_path = os.path.join(img_root, target_dir, city, img_file)
            annot_path = os.path.join(annot_root, target_dir, city, annot_file)
            img = cv2.imread(img_path)
            with open(annot_path) as f:
                annot = eval(f.read())
            print("\n---")
            print(img.shape)
            print(annot)
            w, h = annot["imgWidth"], annot["imgHeight"]
            valid_img = False
            for box in annot["objects"]:
                if box["label"] == "ignore":
                    continue
                draw_xywh(box["bbox"], img, (0, 0, 225))
                draw_xywh(box["bboxVis"], img, (0, 255, 0))
                valid_img = True

            if valid_img:
                cv2.imshow("img", img)
                cv2.waitKey(0)


