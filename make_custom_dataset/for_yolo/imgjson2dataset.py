import os

import cv2
import xmltodict


if __name__ == "__main__":
    root = "/home/daton/Downloads/firedetection"
    img_dir = os.path.join(root, "train")
    label_dir = os.path.join(root, "train")

    out_root = "/home/daton/Downloads/fire detection dataset/train"
    out_img_dir = os.path.join(out_root, "images")
    out_label_dir = os.path.join(out_root, "labels")

    imgs = [x for x in sorted(os.listdir(img_dir)) if (os.path.isfile(os.path.join(img_dir, x))
                                                       and x.endswith(".jpg"))]
    for img_name in imgs:
        img_path = os.path.join(img_dir, img_name)
        annot_path = os.path.join(label_dir, img_name.replace(".jpg", ".xml"))
        with open(annot_path) as f:
            data = f.read()
            annot = xmltodict.parse(data)["annotation"]
        img = cv2.imread(img_path)
        imv = img.copy()

        h, w = img.shape[:2]
        boxes = annot["object"]
        if type(boxes) != list:
            boxes = [boxes]
        for tmp in boxes:
            print(tmp)
            box = tmp["bndbox"]
            xyxy = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
            cv2.rectangle(imv, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), [255, 0, 0], 2)

        cv2.imshow("img", imv)
        cv2.waitKey(0)