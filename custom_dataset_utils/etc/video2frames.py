import argparse
import os
import sys
from pathlib import Path
import cv2

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[2].as_posix(), "yolov5"))
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import increment_path




def main(opt):
    source = opt.source
    out_dir = opt.out_dir
    out_name = opt.out_name
    save = opt.save

    if save:
        save_dir = increment_path(Path(out_dir) / out_name, exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)

    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    ) or source.startswith("/dev/video")
    if webcam:
        dataset = LoadStreams(source, img_size=640, stride=32, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=640, stride=32, auto=True)
        bs = 1

    for path, img, im0s, vid_cap in dataset:
        for i, _ in enumerate(im0s):
            print("\n---")
            print(vid_cap[i].get(cv2.CAP_PROP_FRAME_COUNT))
            print(vid_cap[i].get(cv2.CAP_PROP_POS_FRAMES))
            img = im0s[i]
            cv2.imshow(f"img{i}", im0s[i])
            cv2.waitKey(1)

            if save:
                save_name = f"video_{int(vid_cap[i].get(cv2.CAP_PROP_POS_FRAMES)):05}.png"
                save_path = str(save_dir / save_name)
                print(save_path)
                cv2.imwrite(save_path, img)


def parse_opt():
    parser = argparse.ArgumentParser()

    source = "https://www.youtube.com/watch?v=aQfObI_FAAw"
    source = "https://www.youtube.com/watch?v=668J-hyfJ0E"
    parser.add_argument("--source", type=str, default=source)

    out_dir = "/media/daton/D6A88B27A88B0569/dataset/video2frames"
    parser.add_argument("--out-dir", type=str, default=out_dir)

    out_name = "exp"
    parser.add_argument("--out-name", type=str, default=out_name)

    parser.add_argument("--save", type=bool, default=False)

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
