import os
import time
import zipfile
import shutil

import cv2
from PIL import Image
from tqdm import tqdm


def extract_rgb_frames_onevid(input_root, vid_name, output_root, save=True, vis=False):
    input_vid = os.path.join(input_root, vid_name)
    cap = cv2.VideoCapture(input_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0

    output_dir = os.path.join(output_root, vid_name.split(".")[0])
    if not os.path.isdir(output_dir):
        if save:
            os.makedirs(output_dir)
    else:
        saved_frames = [int(x.split(".")[0].split("_")[-1]) for x in os.listdir(output_dir) if x.endswith(".jpg")]
        start_frame = max(saved_frames)
    print(f"--- {vid_name}")
    print(f"{start_frame} ~ {total_frames}")
    time.sleep(0.1)
    for i in tqdm(range(start_frame, total_frames)):
        t1 = time.time()
        ret, img = cap.read()
        t2 = time.time()
        if not ret:
            time.sleep(0.1)
            print(f"\ntotal {i} images are extracted")
            break
        if vis:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        if save:
            img = Image.fromarray(img[..., ::-1])
            tmp_img_name = f"img_{i + 1:05d}.jpg"
            tmp_img_path = os.path.join(output_dir, tmp_img_name)
            t1 = time.time()
            img.save(tmp_img_path)
            t2 = time.time()
    cap.release()


def extract_rgb_frames(input_root, output_root, ref_root, save=True, vis=False):
    aleady_names = [x.split(".")[0] for x in os.listdir(ref_root) if os.path.isfile(os.path.join(ref_root, x))]
    vid_names = [x for x in os.listdir(input_root) if os.path.isfile(os.path.join(input_root, x))
                 and x.split(".")[0] not in aleady_names]
    print(aleady_names)
    print(vid_names)
    interval = 60
    for i, vid_name in enumerate(vid_names):
        print(f"\n############# {i + 1}  / {len(vid_names)}")
        extract_rgb_frames_onevid(input_root, vid_name, output_root, save, vis)
        shutil.make_archive(os.path.join(output_root, f"{vid_name.split('.')[0]}"), "zip",
                            os.path.join(output_root, vid_name.split(".")[0]))
        shutil.rmtree(os.path.join(output_root, vid_name.split(".")[0]))
        if (i + 1) % interval == 0:
            break


if __name__ == "__main__":
    input_root = "/media/daton/Data/dataset/ava/videos_15min"
    #input_root = "/media/daton/Data/dataset/ava/target_videos_15min"
    vid_name = "-5KQ66BBWC4.mkv"
    vid_name = '-FaXLcSFjUI.mp4'
    #output_root = "/media/daton/SAMSUNG/dataset/ava/rawframes"
    output_root = "/media/daton/Data/dataset/ava/rawframes"
    ref_root = "/media/daton/SAMSUNG/dataset/ava/rawframes"

    #extract_rgb_frames_onevid(input_root, vid_name, output_root, save=True, vis=False)
    extract_rgb_frames(input_root, output_root, ref_root, save=True, vis=False)
