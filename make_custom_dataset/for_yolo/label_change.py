import os
from tqdm import tqdm

target_dir = "/home/daton/Downloads/fire detector.v1i.yolov5pytorch/train/labels"
out_dir = "/home/daton/Downloads/fire detector.v1i.yolov5pytorch/train/labels_edit"

files = sorted(os.listdir(target_dir))
for file in tqdm(files):
    file_path = os.path.join(target_dir, file)
    with open(file_path) as f:
        data = f.readlines()

        edit_data = []
        for d in data:
            if d[0] == "3":
                line = "0" + d[1:]
            elif d[0] == "2":
                line = d
            elif d[0] == "0":
                line = "1" + d[1:]
            elif d[0] == 1:
                line = "3" + d[1:]
            edit_data.append(line)

        str = ""
        for d in edit_data:
            str += d

        out_path = os.path.join(out_dir, file)
        with open(out_path, "w") as o:
            o.write(str)
