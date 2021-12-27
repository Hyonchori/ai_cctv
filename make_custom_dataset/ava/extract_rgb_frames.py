import os

import cv2
from tqdm import tqdm


def extract_rgb_frames_onevid(input_root, vid_name, output_root, save=True, vis=False):
    input_vid = os.path.join(input_root, vid_name)
    cap = cv2.VideoCapture(input_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if save:
        output_dir = os.path.join(output_root, vid_name.split(".")[0])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    print(f"\n--- {vid_name}")
    print(total_frames)
    for i in tqdm(range(total_frames)):
        ret, img = cap.read()
        if not ret:
            print(f"\ntotal {i} images are extracted")
            break
        if vis:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        if save:
            tmp_img_name = f"img_{i + 1:05d}.jpg"
            tmp_img_path = os.path.join(output_dir, tmp_img_name)
            cv2.imwrite(tmp_img_path, img)
    cap.release()


def extract_rgb_frames(input_root, output_root, save=True, vis=False):
    vid_names = [x for x in os.listdir(input_root) if os.path.isfile(os.path.join(input_root, x))]
    for vid_name in vid_names:
        extract_rgb_frames_onevid(input_root, vid_name, output_root, save, vis)


if __name__ == "__main__":
    input_root = "/media/daton/SAMSUNG/dataset/ava/videos_15min"
    vid_name = "-5KQ66BBWC4.mkv"
    output_root = "/media/daton/SAMSUNG/dataset/ava/rawframes_e"

    #extract_rgb_frames_onevid(input_root, vid_name, output_root, save=True, vis=True)
    extract_rgb_frames(input_root, output_root, save=True, vis=False)