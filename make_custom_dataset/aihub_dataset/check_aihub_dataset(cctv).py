import os
import json
import shutil
import time

import cv2
from tqdm import tqdm


def show_target_idx_num(root, target, idx, num, view=True, rm_short=True, resize=(1280, 720)):
    img_root_dir = os.path.join(root, target, f"[원천]{target}_{idx}")
    label_dir = os.path.join(root, target, f"[라벨]{target}_{idx}")
    img_dir = os.path.join(img_root_dir, str(num))
    annot_file = os.path.join(label_dir, f"annotation_{num}.json")
    vid_path = ""
    with open(annot_file) as f:
        annot = json.load(f)
    if len(annot["frames"]) >= 150:
        if view:
            for frame in annot["frames"]:
                tmp_annots = frame["annotations"]
                img_path = os.path.join(img_dir, frame["image"])
                img = cv2.imread(img_path)
                for l in tmp_annots:
                    box = l["label"]
                    name = l["category"]["code"]
                    xyxy = box2xyxy(box)
                    #cv2.putText(img, name, (xyxy[0], xyxy[1]), 0, 2, (0, 255, 0), 2)
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                img = cv2.resize(img, dsize=resize)
                cv2.imshow(f"{num} - {len(annot['frames'])}", img)
                cv2.waitKey(1)
            cv2.destroyAllWindows()
    else:
        if rm_short:
            shutil.rmtree(img_dir, ignore_errors=True)
            os.remove(annot_file)


def box2xywh(box):
    xywh = [box["x"], box["y"], box["width"], box["height"]]
    return xywh


def xywh2xyxy(xywh):
    xyxy = [xywh[0],
            xywh[1],
            xywh[0] + xywh[2],
            xywh[1] + xywh[3]]
    return xyxy


def box2xyxy(box):
    return xywh2xyxy(box2xywh(box))


def show_target_idx_nums(root, target, idx):
    img_root_dir = os.path.join(root, target, f"[원천]{target}_{idx}")
    nums = [x for x in sorted(os.listdir(img_root_dir)) if (os.path.isdir(os.path.join(img_root_dir, x))
                                                            and x.isdigit())]
    print(f"\n--- Processing {img_root_dir}")
    time.sleep(0.5)
    for num in tqdm(nums):
        show_target_idx_num(root, target, idx, num, view=True, rm_short=False)


if __name__ == "__main__":
    root = "/media/daton/SAMSUNG/지하철 역사 내 CCTV 이상행동 영상/Training"
    target = "개집표기 무단진입"
    idx = 7


    num = 2112680
    #show_target_idx_num(root, target, idx, num)
    show_target_idx_nums(root, target, idx)