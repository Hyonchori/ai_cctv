import os

import cv2
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_clf_dirs(data_set, train_root, out_root, out_dir="train"):
    for img_name, img_label in zip(data_set["file_name"], data_set["label"]):
        img_path = os.path.join(train_root, img_name)
        img = cv2.imread(img_path)

        label_dir = os.path.join(out_root, out_dir, str(img_label))
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        out_path = os.path.join(label_dir, img_name)
        cv2.imwrite(out_path, img)


def make_train_valid(train_root, train_csv, out_root, save=False):
    data = pd.read_csv(train_csv)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, valid_idx in split.split(data, data["label"]):
        train_set = data.loc[train_idx]
        valid_set = data.loc[valid_idx]

    if save:
        make_clf_dirs(train_set, train_root, out_root, "train")
        make_clf_dirs(valid_set, train_root, out_root, "valid")


if __name__ == "__main__":
    root = "/home/daton/Downloads/dataset"
    train_root = os.path.join(root, "train")
    test_root = os.path.join(root, "test")
    csv_path = os.path.join(root, "train_data.csv")

    out_root = "/media/daton/D6A88B27A88B0569/dataset/mnist/basic"

    make_train_valid(train_root, csv_path, out_root, save=True)