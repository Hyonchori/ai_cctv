import os
from tqdm import tqdm

target_dir = "/home/daton/Downloads/edit_military.v1i.yolov5pytorch/train/labels"
out_dir = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman_military/labels/train"

files = sorted(os.listdir(target_dir))
for file in tqdm(files):
    file_path = os.path.join(target_dir, file)
    with open(file_path) as f:
        data = f.readlines()

        edit_data = []
        for d in data:
            if d[0] == "2":
                line = "0" + d[1:]
            elif d[0] == "1":
                line = d
            elif d[0] == "0":
                line = "2" + d[1:]
            edit_data.append(line)


        str = ""
        for d in edit_data:
            str += d

        out_path = os.path.join(out_dir, file)
        with open(out_path, "w") as o:
            o.write(str)
