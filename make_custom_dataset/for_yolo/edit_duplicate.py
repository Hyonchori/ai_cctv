import os
from tqdm import tqdm


def check_duplicate(label_dir, edit=False):
    labels = sorted(os.listdir(label_dir))
    corrupt_count = 0
    for l in tqdm(labels):
        label_path = os.path.join(label_dir, l)
        is_corrupt = False
        with open(label_path) as f:
            data = f.readlines()
            data_edit = list(set(data))
            if len(data) != len(data_edit):
                corrupt_count += 1
                is_corrupt = True

        if is_corrupt and edit:
            txt = ""
            for box in data_edit:
                txt += box
            with open(label_path, "w") as f:
                f.write(txt)
    print(corrupt_count)


if __name__ == "__main__":
    label_dir = "/home/daton/Downloads/fire detector.v1i.yolov5pytorch/valid/labels"
    check_duplicate(label_dir, True)