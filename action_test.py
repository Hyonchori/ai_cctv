import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.utils.augmentations import letterbox

from deep_sort_pytorch.utils.parser import get_config as get_deepsort_cfg
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.downloads import attempt_download

from stdet import StdetPredictor
from mmcv import Config as get_stdet_cfg


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
    yolo_save_crop = opt.yolo_save_crop

    deepsort_cfg = get_deepsort_cfg()
    deepsort_cfg.merge_from_file(opt.deepsort_cfg)
    deepsort_weights = opt.deepsort_weights

    stdet_cfg = get_stdet_cfg.fromfile(opt.stdet_cfg)
    stdet_cfg.merge_from_dict(opt.stdet_cfg_options)

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

    attempt_download(deepsort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs
    deepsort_model_list = [DeepSort(deepsort_cfg.DEEPSORT.REID_CKPT,
                                    max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
                                    min_confidence=deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                                    max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=deepsort_cfg.DEEPSORT.MAX_AGE, n_init=deepsort_cfg.DEEPSORT.N_INIT,
                                    nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True) for _ in range(bs)]

    stdet_model = StdetPredictor(
        config=stdet_cfg,
        checkpoint=opt.stdet_weights,
        device=device,
        score_thr=opt.stdet_action_score_thr,
        label_map_path=opt.stdet_label_map_path
    )

    # Run inference
    if device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters())))
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        print("\n---")
        img = torch.from_numpy(img).to(device)
        img = img.half() if yolo_half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        t1 = time_sync()
        pred = yolo_model(img)[0]

        # NMS
        pred = non_max_suppression(pred, yolo_conf_thr, yolo_iou_thr, yolo_target_classes, max_det=yolo_max_det)
        t2 = time_sync()

        # Process predictions
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)



def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5l_crowdhuman_v2.pt"
    #yolo_weights = "yolov5x.pt"
    parser.add_argument("--yolo_weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img", "--iyolo-mg-size", type=int, default=[640])
    parser.add_argument("--yolo-conf_thr", "--yolo-conf_thres", type=float, default=0.55)
    parser.add_argument("--yolo-iou-thr", "--yolo-iou-thres", type=float, default=0.5)
    parser.add_argument("--yolo-max-det", type=int, default=1000)
    parser.add_argument("--yolo-target-classes", default=None, nargs="+", type=int)
    parser.add_argument("--yolo-hide-labels", default=False, action="store_true")
    parser.add_argument("--yolo-hide-conf", default=False, action="store_true")
    parser.add_argument("--yolo-half", default=False, action="store_true")
    parser.add_argument("--yolo-save-crop", default=False, action="store_true")

    parser.add_argument("--deepsort-cfg", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--deepsort-weights", type=str, default="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

    parser.add_argument("--stdet-cfg", default=("../mmaction2/configs/detection/ava/"
                                                "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py"))
    parser.add_argument("--stdet-weights", default=('https://download.openmmlab.com/mmaction/detection/ava/'
                                                    'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                                                    'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                                                    '_20201217-16378594.pth'))
    parser.add_argument("--stdet-action-score-thr", type=float, default=0.4)
    parser.add_argument("--stdet-label-map-path", default="../mmaction2/tools/data/ava/label_map.txt")
    parser.add_argument("--stdet-cfg-options", default={})

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    #source = "source.txt"
    #source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/test/MOT17-03-DPM/img1"
    #source = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/이미지/image_action_45/image_45-1/45-1/45-1_001-C01"
    #source = "https://www.youtube.com/watch?v=-gSOi6diYzI"
    #source = "https://www.youtube.com/watch?v=gwavBeK4H1Q"
    #source = "0"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--save-vid", type=bool, default=True)
    parser.add_argument("--show-vid", type=bool, default=True)

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