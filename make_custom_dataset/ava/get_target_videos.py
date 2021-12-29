import os

import cv2
import pandas


def visualize_video(annot_root, video_root, video_name):
    pass

if __name__ == "__main__":
    annot_root = "/media/daton/Data/dataset/ava/annotations"
    train_file = os.path.join(annot_root, "ava_train_v2.1.csv")
    valid_file = os.path.join(annot_root, "ava_val_v2.1.csv")

    video_root = "/media/daton/Data/dataset/ava/videos_15min"
    video_name = "-5KQ66BBWC4.mkv"

    train_annot = pandas.read_csv(train_file).to_numpy()
    print(train_annot)
    valid_annot = pandas.read_csv(valid_file).to_numpy()
    print(valid_annot)

    train_vids = list(set(train_annot[:, 0]))
    print(len(train_vids))
    valid_vids = list(set(valid_annot[:, 0]))
    print(len(valid_vids))