import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 480))


def make_labels(annot_root, img_root, make_root, vis=False, save=False, iou_thr=0.54):
    with open(annot_root) as f:
        data = f.readlines()
        num_body = 0
        num_head = 0
        num_bodyu = 0
        num_headu = 0
        for d in tqdm(data):
            d_dict = eval(d)
            make_path = os.path.join(make_root, d_dict["ID"] + ".txt")
            img_path = os.path.join(img_root, d_dict["ID"] + ".jpg")
            img = cv2.imread(img_path)
            imgc = img.copy()
            h, w, _ = img.shape

            txt = ""
            gtboxes = d_dict["gtboxes"]
            gtboxes = [x for x in gtboxes if x["tag"] == "person"]

            too_big_head = False
            for gtbox in gtboxes:
                use_head = 1
                if "head_attr" in gtbox.keys():
                    if "ignore" in gtbox["head_attr"].keys():
                        if gtbox["head_attr"]["ignore"] == 1:
                            use_head = 2
                            num_headu += 1
                            continue

                if "ignore" in gtbox["extra"].keys():
                    if gtbox["extra"]["ignore"] == 1:
                        use_head = 3
                        num_bodyu += 1
                        continue

                fbox = gtbox["fbox"]  # full body including occlusion
                vbox = gtbox["vbox"]  # visible body in image
                hbox = gtbox["hbox"]  # head in image

                fxyxy_e = xywh2xyxye(fbox, w, h)
                hxyxy_e = xywh2xyxye(hbox, w, h)

                iou = compare_iou(fxyxy_e, hxyxy_e)
                if iou >= iou_thr:
                    print(iou)
                    use_head = 4
                    too_big_head = True

                body_txt = ""
                if check_xyxy(fxyxy_e) and use_head == 1:
                    draw_xyxy(fxyxy_e, img, (225, 0, 0))
                    body_cpwh = xyxy2cpwhn(fxyxy_e, w, h)
                    body_txt = f"0 {body_cpwh[0]} {body_cpwh[1]} {body_cpwh[2]} {body_cpwh[3]}\n"
                    num_body += 1

                head_txt = ""
                if check_xyxy(hxyxy_e):
                    num_head += 1
                    head_cpwh = xyxy2cpwhn(hxyxy_e, w, h)
                    head_img = imgc[hxyxy_e[1]: hxyxy_e[3], hxyxy_e[0]: hxyxy_e[2]]
                    face = app.get(head_img)
                    if len(face) >= 1:
                        if face[0]["det_score"] >= 0.65:
                            head_idx = 2
                        else:
                            head_idx = 1
                    else:
                        head_idx = 1
                    head_txt = f"{head_idx} {head_cpwh[0]} {head_cpwh[1]} {head_cpwh[2]} {head_cpwh[3]}\n"
                    if use_head == 1:
                        pass
                        if vis: draw_xyxy(hxyxy_e, img, (0, 0, 225))
                    elif use_head == 2:
                        print("")
                        print(f"use_head: 2 => {gtbox}")
                        print(hxyxy_e)
                        if vis: draw_xyxy(hxyxy_e, img, (0, 225, 0))
                    elif use_head == 3:
                        print("")
                        print(f"use_gead: 3 => {gtbox}")
                        if vis: draw_xyxy(hxyxy_e, img, (0, 225, 225))
                    elif use_head == 4:
                        if vis:
                            print("")
                            print(f"use_gead: 4 => {gtbox}")
                            draw_xyxy(hxyxy_e, img, (225, 225, 0))
                txt += body_txt
                txt += head_txt

            if save:
                with open(make_path, "w") as s:
                    s.write(txt)

            if vis and too_big_head:
                cv2.imshow("img", img)
                cv2.waitKey(0)

        print("")
        print(f"num body: {num_body}")
        print(f"num head: {num_head}")
        print(f"num body unused: {num_bodyu}")
        print(f"num head unused: {num_headu}")


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


def xywh2xyxye(box, w, h):
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


def compare_iou(xyxy1, xyxy2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = xyxy1[0], xyxy1[1], xyxy1[2], xyxy1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = xyxy2[0], xyxy2[1], xyxy2[2], xyxy2[3]
    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * \
            max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    a1, a2 = w1 * h1, w2 * h2

    iou = inter / union
    return iou


def xyxy2cpwhn(xyxy, w, h):
    cpwhn = [round(((xyxy[0] + xyxy[2]) / 2 / w), 6),
             round(((xyxy[1] + xyxy[3]) / 2 / h), 6),
             round(((xyxy[2] - xyxy[0]) / w), 6),
             round(((xyxy[3] - xyxy[1]) / h), 6)]
    return cpwhn

if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman"
    train_img_path = os.path.join(root, "train", "images")
    valid_img_path = os.path.join(root, "valid", "images")
    label_dir_name = "labels_new"

    train_annot_path = os.path.join(root, "annotation_train.odgt")
    train_make_root = os.path.join(root, "train", "labels")
    if not os.path.isdir(train_make_root):
        os.makedirs(train_make_root)

    valid_annot_path = os.path.join(root, "annotation_val.odgt")
    valid_make_root = os.path.join(root, "valid", "labels")
    if not os.path.isdir(valid_make_root):
        os.makedirs(valid_make_root)

    make_labels(train_annot_path, train_img_path, train_make_root, vis=False, save=True)
    make_labels(valid_annot_path, valid_img_path, valid_make_root, vis=False, save=True)