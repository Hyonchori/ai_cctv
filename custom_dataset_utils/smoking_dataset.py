
import cv2
import os
import json
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np


class SmokingDatasetFromOneVideo(torch.utils.data.Dataset):
    def __init__(self, img_dir, annot_path):
        img_names = sorted(os.listdir(img_dir))
        self.imgs = [os.path.join(img_dir, img_name) for img_name in img_names]
        self.annot_path = annot_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img0 = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        if os.path.isfile(self.annot_path):
            with open(self.annot_path, "r") as f:
                data = json.load(f)
                annotations = data["annotations"]
                annot = annotations[idx]
                bbox = annot["bbox"]
                keypoint = annot["keypoints"]

        return img0, bbox, keypoint


class SmokingDatasetFromOneDir(torch.utils.data.Dataset):
    def __init__(self,
                 video_root: str,
                 video_num: int,
                 video_idx: int,
                 annot_root: str,
                 _2d_or_3d: str="2d",):
        video_dir = os.path.join(video_root, f"image_action_{video_num}",
                                 f"image_{video_num}-{video_idx}", f"{video_num}-{video_idx}")
        if _2d_or_3d not in ["2d", "3d"]:
            raise Exception(f"'{_2d_or_3d}' should be one of ['2d', '3d']")
        if _2d_or_3d == "2d":
            annot_dir = os.path.join(annot_root, "Annotation_2D_tar", "2D", f"{video_num}-{video_idx}")
        elif _2d_or_3d == "3d":
            annot_dir = os.path.join(annot_root, "Annotation_3D_tar", "3D", f"{video_num}-{video_idx}")

        if not os.path.isdir(video_dir):
            raise Exception(f"'{video_dir}' is not a directory!")
        videos = sorted(os.listdir(video_dir))
        self.videos_path = [os.path.join(video_dir, video) for video in videos]

        if not os.path.isdir(annot_dir):
            raise Exception(f"'{annot_dir}' is not a directory")
        annots = sorted(os.listdir(annot_dir))
        self.annots_path = [os.path.join(annot_dir, annot) for annot in annots
                            if not annot.startswith(".")]

        self.datasets = [SmokingDatasetFromOneVideo(video, annot) for video, annot
                         in zip(self.videos_path, self.annots_path)]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.datasets[idx]


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/"
    video_root = os.path.join(root, "이미지")
    annot_root = os.path.join(root, "annotation")
    datasets = SmokingDatasetFromOneDir(video_root=video_root,
                                        annot_root=annot_root,
                                        video_num=45,
                                        video_idx=2)