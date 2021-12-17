import os
import json
import shutil
import time
import zipfile

import cv2
from tqdm import tqdm


def unzip_and_remove(root, target):
    target_dir = os.path.join(root, target)
    zips = {}
    for file in os.listdir(target_dir):
        if not file.endswith(".zip"):
            continue
        file_idx = int(file.split(".")[0][-1])
        if file_idx in zips:
            zips[file_idx].append(file)
        else:
            zips[file_idx] = [file]

    for idx, files in zips.items():
        for file in files:
            file_path = os.path.join(target_dir, file)
            print(file_path)
            tmp_dir_path = file_path.replace(".zip", "")
            if not os.path.isdir(tmp_dir_path):
                os.mkdir(tmp_dir_path)
            if len(os.listdir(tmp_dir_path)) == 0:
                fantasy_zip = zipfile.ZipFile(file_path)
                fantasy_zip.extractall(tmp_dir_path)

            img_dirs = [x for x in os.listdir(tmp_dir_path) if (os.path.isdir(os.path.join(tmp_dir_path, x))
                                                                and x.isdigit())]
            print(img_dirs)
        break


if __name__ == "__main__":
    root = "/media/daton/SAMSUNG1/지하철 역사 내 CCTV 이상행동 영상/Training"
    target = "에스컬레이터 전도"

    unzip_and_remove(root, target)
