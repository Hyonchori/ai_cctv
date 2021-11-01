import os
from tqdm import tqdm

body_dir = "/media/daton/D6A88B27A88B0569/datas련et/military/dataset3/train/labels"
head_dir = "/home/daton/PycharmProjects/pythonProject/yolov5/runs/detect/exp16/labels"
out_dir = "/media/daton/D6A88B27A88B0569/datas련et/military/dataset3/train/labels_edit"

files = os.listdir(body_dir)
for file in tqdm(files):
    body_path = os.path.join(body_dir, file)
    head_path = os.path.join(head_dir, file)
    out_path = os.path.join(out_dir, file)

    with open(body_path) as b:
        body = b.readlines()

    if os.path.isfile(head_path):
        with open(head_path) as h:
            head = h.readlines()
    else:
        head = []

    for line in head:
        if not line.startswith("0"):
            body.append(line)

    str = ""
    for line in body:
        str += line

    with open(out_path, "w") as f:
        f.write(str)