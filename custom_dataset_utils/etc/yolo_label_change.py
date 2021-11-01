import os
from tqdm import tqdm

in_dir = "/home/daton/PycharmProjects/pythonProject/yolov5/runs/detect/exp11/labels"
out_dir = "/home/daton/PycharmProjects/pythonProject/yolov5/runs/detect/exp11/labels_1"

in_labels = os.listdir(in_dir)
for file in tqdm(in_labels):
    file_path = os.path.join(in_dir, file)
    with open(file_path) as f:
        lines = f.readlines()
        lines = ["1" + line[1:] for line in lines]
        str = ""
        for line in lines:
            str += line

    out_path = os.path.join(out_dir, file)
    with open(out_path, "w") as f:
        f.write(str)