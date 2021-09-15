
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
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "hrnet", "custom_lib"))
from hrnet.custom_lib import hrnet_models
from hrnet.custom_lib.config import cfg
from hrnet.custom_lib.config import update_config
from hrnet.custom_lib.hrnet_utils.inference_utils import draw_pose, box_to_center_scale, \
    get_pose_estimation_prediction


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

    update_config(cfg, opt)
    hrnet_vis_thr = opt.hrnet_vis_thr

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

    hrnet_model = eval(f"hrnet_models.{cfg.MODEL.NAME}.get_pose_net")(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        hrnet_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    hrnet_model = torch.nn.DataParallel(hrnet_model, device_ids=cfg.GPUS)
    hrnet_model.to(device)
    hrnet_model.eval()

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
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
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

            p = Path(p)
            save_path = str(save_dir / p.name)
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if yolo_save_crop else im0
            img_pose = im0.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if yolo_hide_labels else (names[c] if yolo_hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    if yolo_save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # Keypoint estimation
                    if c == 0:
                        box = [(xyxy[0].item(), xyxy[1].item()), (xyxy[2].item(), xyxy[3].item())]
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        kp_preds, kp_confs = get_pose_estimation_prediction(hrnet_model, img_pose, center, scale, cfg)
                        if len(kp_preds) >= 1:
                            for kpt, kpc in zip(kp_preds, kp_confs):
                                draw_pose(kpt, im0, kpc, hrnet_vis_thr)


            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s")

            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)

            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += ".mp4"
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                vid_writer[i].write(im0)

def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5l_crowdhuman_v2.pt"
    parser.add_argument("--yolo_weights", nargs="+", type=str, default=yolo_weights)

    parser.add_argument("--yolo-imgsz", "--yolo-img", "--iyolo-mg-size", type=int, default=[640])
    parser.add_argument("--yolo-conf_thr", "--yolo-conf_thres", type=float, default=0.75)
    parser.add_argument("--yolo-iou-thr", "--yolo-iou-thres", type=float, default=0.4)
    parser.add_argument("--yolo-max-det", type=int, default=1000)
    parser.add_argument("--yolo-target-classes", nargs="+", type=int)
    parser.add_argument("--yolo-hide-labels", default=False, action="store_true")
    parser.add_argument("--yolo-hide-conf", default=False, action="store_true")
    parser.add_argument("--yolo-half", default=False, action="store_true")
    parser.add_argument("--yolo-save-crop", default=False, action="store_true")

    hrnet_cfg = "inference-config.yaml"
    parser.add_argument("--hrnet-cfg", type=str, default=hrnet_cfg)
    parser.add_argument("--hrnet-opts", default=[])
    parser.add_argument("--hrnet-modelDir", default="")
    parser.add_argument("--hrnet-logDir", default="")
    parser.add_argument("--hrnet-dataDir", default="")
    parser.add_argument("--hrnet-vis-thr", type=float, default=0.6)

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "0"
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
