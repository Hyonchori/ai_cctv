
import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(opt):
    source = opt.source
    device = opt.device
    project = opt.project
    name = opt.name

    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_max_det = opt.yolo_max_det
    yolo_target_classes = opt.yolo_target_classes
    yolo_hide_labels = opt.yolo_hide_labels
    yolo_hide_conf = opt.yolo_hide_conf
    yolo_half = opt.yolo_half

    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    # Directories result will be saved
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    yolo_half &= device.type != "cpu"

    # Load model
    stride, names = 64, [f"class{i}" for i in range(1000)]
    yolo_model = attempt_load(yolo_weights, map_location=device)
    stride = int(yolo_model.stride.max())
    names = yolo_model.module.names if hasattr(yolo_model, "module") else yolo_model.names
    if yolo_half:
        yolo_model.half()
    yolo_imgsz = check_img_size(yolo_imgsz, s=stride)
    ascii = is_ascii(names)

    # DataLoader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters())))


def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5l_crowdhuman_v2.pt"
    parser.add_argument("--yolo_weights", nargs="+", type=str, default=yolo_weights)

    parser.add_argument("--yolo-imgsz", "--yolo-img", "--iyolo-mg-size", type=int, default=[640])
    parser.add_argument("--yolo-conf_thr", "--yolo-conf_thres", type=float, default=0.7)
    parser.add_argument("--yolo-iou-thr", "--yolo-iou-thres", type=float, default=0.5)
    parser.add_argument("--yolo-max-det", type=int, default=1000)
    parser.add_argument("--yolo-target-classes", nargs="+", type=int)
    parser.add_argument("--yolo-hide-labels", default=False, action="store_true")
    parser.add_argument("--yolo-hide-conf", default=False, action="store_true")
    parser.add_argument("--yolo-half", default=False, action="store_true")

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(requirements="yolov5/requirements.txt", exclude=("tensorboard", "thop"))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
