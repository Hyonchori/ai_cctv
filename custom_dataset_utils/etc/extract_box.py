import os
import cv2
from tqdm import tqdm
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(128)
])

in_root = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman"
out_root = "/media/daton/D6A88B27A88B0569/dataset/military_civil2"
box_dirs = [os.path.join(in_root, "labels", trainval) for trainval in ["train", "valid"]]
print(box_dirs)
box_paths = {}
for dir in box_dirs:
    for file in os.listdir(dir):
        box_paths[file] = os.path.join(dir, file)

#target_dir = os.path.join(in_root, "images_only_military")
target_dir = os.path.join(in_root, "images", "train")
target_names = os.listdir(target_dir)
for i, file in enumerate(tqdm(target_names)):
    img_path = os.path.join(target_dir, file)
    box_name = file.replace(".jpg", ".txt", -1)
    box_path = box_paths[box_name]

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    with open(box_path) as b:
        boxes = [x.replace("\n", "") for x in b.readlines()]
    for j, box in enumerate(boxes):
        box_split = box.split()
        cls = int(box_split[0])
        if cls != 0:
            continue
        cx = float(box_split[1])
        cy = float(box_split[2])
        w = float(box_split[3])
        h = float(box_split[4])

        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        box_img = img[y1: y2, x1: x2]
        if 0 in box_img.shape:
            continue

        #out_path = os.path.join(out_root, "military", f"{i}-{j}.jpg")
        out_path = os.path.join(out_root, "train", "civil", f"{i}-{j}.jpg")
        cv2.imwrite(out_path, box_img)



