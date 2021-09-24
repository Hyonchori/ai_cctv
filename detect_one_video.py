import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
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

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.downloads import attempt_download

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "hrnet", "custom_lib"))
from hrnet.custom_lib import hrnet_models
from hrnet.custom_lib.config import cfg
from hrnet.custom_lib.config import update_config
from hrnet.custom_lib.hrnet_utils.inference_utils import draw_pose, box_to_center_scale, \
    get_pose_estimation_prediction_from_batch, get_pose_estimation_prediction, transform

from sensing_mode import RoI, SenseTrespassing, SenseLoitering


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

    deepsort_cfg = get_config()
    deepsort_cfg.merge_from_file(opt.deepsort_cfg)
    deepsort_weights = opt.deepsort_weights

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

    attempt_download(deepsort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')

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
    deepsort_model_list = [DeepSort(deepsort_cfg.DEEPSORT.REID_CKPT,
                           max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
                           min_confidence=deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                           max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                           max_age=deepsort_cfg.DEEPSORT.MAX_AGE, n_init=deepsort_cfg.DEEPSORT.N_INIT,
                           nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
                           use_cuda=True) for _ in range(bs)]


    roi = RoI()
    roi = SenseTrespassing(colors=colors)
    target_fps = dataset.fps if hasattr(dataset, "fps") else 15
    target_fps = target_fps[0] if isinstance(target_fps, list) else target_fps
    roi = SenseLoitering(colors=colors,
                         fps=target_fps,
                         max_buffer_size=target_fps * 15,
                         time_thr=10)

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

        if roi.img_size == None:
            if len(im0s[0].shape) == 3:
                roi.img_size = im0s[0].shape
            else:
                roi.img_size = im0s.shape
            roi.ref_img = np.zeros(roi.img_size, np.uint8)
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
            #save_path = str(save_dir / "video")
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if yolo_save_crop else im0
            img_pose = im0.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            bodies = []
            faces = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #roi.update(det)
                '''for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    if c == 0:
                        label = f"{names[c]} {conf:.2f}"
                        label = f"{names[c]}"
                        annotator.box_label(xyxy, label, color=colors(c, True))'''

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Pass detection to deepsort
                clss = det[:, 5]
                person_idx = clss == 0
                xywhs = xyxy2xywh(det[:, 0:4])[person_idx]
                confs = det[:, 4][person_idx]
                clss = clss[person_idx]
                outputs = deepsort_model_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                roi.update(outputs)

                # Draw visualization
                cropped_person_batch = []
                person_centers = []
                person_scales = []
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(reversed(outputs), reversed(confs))):
                        xyxy = output[0: 4]
                        id = int(output[4])
                        cls = output[5]
                        c = int(cls)
                        label = f"{id} {names[c]} {conf:.2f}"
                        label = f"{names[c]}"
                        #annotator.box_label(xyxy, label, color=colors(id, True))

                        '''if yolo_save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                        # Append bodies and faces
                        if c == 0:
                            bodies.append([*xyxy, conf])
                        elif c >= 1:
                            faces.append([*xyxy, conf])

                        # Keypoint estimation
                        if c == 0:
                            box = [(xyxy[0].item(), xyxy[1].item()), (xyxy[2].item(), xyxy[3].item())]
                            person_crop = img_pose[int(box[0][1]): int(box[1][1]),
                                          int(box[0][0]): int(box[1][0])]
                            person_crop_lb, ratio, _ = letterbox(person_crop,
                                                                 (cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]),
                                                                 auto=False)
                            if 0 in ratio:
                                continue
                            center = [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2]
                            re_scale = [1 / ratio[0] * 1.25 * 1.15, 1 / ratio[1] * 1.25]

                            cropped_person_batch.append(transform(person_crop_lb).unsqueeze(0))
                            person_centers.append(center)
                            person_scales.append(re_scale)'''
                    '''if cropped_person_batch:
                        cropped_person_batch = torch.cat(cropped_person_batch)
                        kp_preds, kp_confs = get_pose_estimation_prediction_from_batch(hrnet_model,
                                                                                       cropped_person_batch,
                                                                                       person_centers,
                                                                                       person_scales,
                                                                                       cfg)
                        for kp_pred, kp_conf in zip(kp_preds, kp_confs):
                            for kpt, kpc in zip(kp_pred, kp_conf):
                                draw_pose(kpt, im0, kpc, hrnet_vis_thr)'''

            else:
                deepsort_model_list[i].increment_ages()
                roi.update([])

            # Tracking
            #print(len(bodies), len(faces))

            # Print time (inference + NMS)
            t3 = time_sync()
            print(f"{s}Done. ({t3 - t1:.3f}s)")

            # Stream results
            im0 = annotator.result()
            if opt.show_vid:
                #cv2.imshow(str(p), im0)
                im0 = roi.imshow(im0)
                cv2.waitKey(1)

            if opt.save_vid:
                if dataset.mode == "imagae":
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
                    print("???")
                    vid_writer[i].write(im0)

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

    hrnet_cfg = "inference-config.yaml"
    #hrnet_cfg = "hrnet_config.yaml"
    parser.add_argument("--hrnet-cfg", type=str, default=hrnet_cfg)
    parser.add_argument("--hrnet-opts", default=[])
    parser.add_argument("--hrnet-modelDir", default="")
    parser.add_argument("--hrnet-logDir", default="")
    parser.add_argument("--hrnet-dataDir", default="")
    parser.add_argument("--hrnet-vis-thr", type=float, default=0.6)

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    #source = "source.txt"
    #source = "http://211.254.214.79:4980/vod/2021/07/16/3-9_2_171/index.m3u8"
    #source = "rtmp://211.254.214.79:4988/CH/CH-0001-zzl5qcmgxg"
    #source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/test/MOT17-03-DPM/img1"
    #source = "/home/daton/Downloads/bandicam 2021-09-24 05-22-34-452.mp4"
    #source = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/이미지/image_action_45/image_45-2/45-2/45-2_001-C02"
    #source = "https://www.youtube.com/watch?v=-gSOi6diYzI"
    #source = "https://www.youtube.com/watch?v=gwavBeK4H1Q"
    #source = "0"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect_one_video")
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