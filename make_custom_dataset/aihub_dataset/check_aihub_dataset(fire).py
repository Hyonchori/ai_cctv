import os
import json

import cv2
from tqdm import tqdm


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()
labels = ["black smoke", "gray smoke", "white smoke", "fire", "cloud", "fog", "light", "sun light", "swing1", "swing2", "nomatter"]


def show_target_idx(root, target, idx, resize=(1280, 720)):
    img_dir = os.path.join(root, f"[원천]{target}_{idx}")
    label_dir = os.path.join(root, f"[라벨]{target}")
    imgs = [x for x in sorted(os.listdir(img_dir)) if x.endswith(".jpg")]
    for img_name in imgs:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        label_name = img_name.replace(".jpg", ".json")
        label_path = os.path.join(label_dir, label_name)
        with open(label_path) as f:
            annot = json.load(f)
        tmp_annots = annot["annotations"]
        for l in tmp_annots:
            cls = int(l["class"])
            color = colors(cls, True)
            if "box" not in l:
                continue
            xyxy = l["box"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{cls - 1}: {labels[cls - 1]}", (xyxy[0], xyxy[1] - 5), font, 1, color, 2)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        img = cv2.resize(img, dsize=resize)
        cv2.imshow("img", img)
        cv2.waitKey(0)


def make_yolov5_dataset_sep(root, target, idx, save_dir, save=False, train=True):
    if train:
        img_dir = os.path.join(root, f"[원천]{target}_{idx}") if idx else os.path.join(root, f"[원천]{target}")
        label_dir = os.path.join(root, f"[라벨]{target}")
    else:
        img_dir = os.path.join(root, f"[원천]{target}{idx}") if idx else os.path.join(root, f"[원천]{target}")
        label_dir = os.path.join(root, f"[라벨]1.{target}")
    save_image_path = os.path.join(save_dir, "images")
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    save_label_path = os.path.join(save_dir, "labels")
    if not os.path.exists(save_label_path):
        os.mkdir(save_label_path)

    imgs = [x for x in sorted(os.listdir(img_dir)) if x.endswith(".jpg")]
    for img_name in tqdm(imgs):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        label_name = img_name.replace(".jpg", ".json")
        label_path = os.path.join(label_dir, label_name)
        if not os.path.isfile(label_path):
            continue
        with open(label_path, encoding="utf-8-sig") as f:
            annot = json.load(f)
        img_info = annot["image"]
        w, h = img_info["resolution"]
        tmp_annots = annot["annotations"]
        txt = ""
        for l in tmp_annots:
            cls = int(l["class"]) - 1
            if "box" not in l:
                continue
            xyxy = l["box"]
            cpwhn = xyxy2cpwhn(xyxy, w, h)
            txt += f"{cls} {cpwhn[0]} {cpwhn[1]} {cpwhn[2]} {cpwhn[3]}\n"
        tmp_save_image_path = os.path.join(save_image_path, img_name)
        tmp_save_label_path = os.path.join(save_label_path, img_name.replace(".jpg", ".txt"))
        if save:
            cv2.imwrite(tmp_save_image_path, img)
            with open(tmp_save_label_path, "w") as f:
                f.write(txt)


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round(int(xyxy[0] + xyxy[2]) / 2 / w, 6),
             round(int(xyxy[1] + xyxy[3]) / 2 / h, 6),
             round(int(xyxy[2] - xyxy[0]) / w, 6),
             round(int(xyxy[3] - xyxy[1]) / h, 6)]
    return cpwhn


def make_yolov5_dataset(root, target, save_dir, save=False, train=True):
    if train:
        target_indices = [x.split("_")[-1] for x in os.listdir(root) if f"[원천]{target}" in x]
    else:
        target_indices = [x[-1] for x in os.listdir(root) if f"[원천]{target}" in x]
    target_indices = [int(x) if x.isdigit() else "" for x in target_indices]
    for idx in target_indices:
        make_yolov5_dataset_sep(root, target, idx, save_dir, save, train)


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/화재 발생 예측 영상/Validation"
    target = "화재씬"
    idx = 2

    #show_target_idx(root, target, idx)

    save_dir = root
    #make_yolov5_dataset_sep(root, target, idx, save_dir, save=True)
    make_yolov5_dataset(root, target, save_dir, save=True, train=False)