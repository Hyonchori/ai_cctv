
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from custom_dataset.smoking_dataset import SmokingDatasetFromOneDir

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh, increment_path, \
    set_logging, check_img_size, is_ascii
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "hrnet", "custom_lib"))
from hrnet.custom_lib import hrnet_models
from hrnet.custom_lib.config import cfg
from hrnet.custom_lib.config import update_config
from hrnet.custom_lib.hrnet_utils.inference_utils import draw_pose, box_to_center_scale, \
    get_pose_estimation_prediction_from_batch, transform


@torch.no_grad()
def main(opt):
    # Load smoking dataset
    root = opt.dataset_root
    video_root = os.path.join(root, "이미지")
    annot_root = os.path.join(root, "annotation")
    datasets = SmokingDatasetFromOneDir(video_root=video_root,
                                        annot_root=annot_root,
                                        video_num=45,
                                        video_idx=2)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 for detection
    set_logging()
    device = select_device(opt.device)
    yolo_model = attempt_load(opt.yolo_weights, map_location=device)
    stride = int(yolo_model.stride.max())
    names = yolo_model.module.names if hasattr(yolo_model, "module") else yolo_model.names
    ascii = is_ascii(names)
    yolo_imgsz = check_img_size(opt.yolo_imgsz, s=stride)

    # Load DeepSORT for tracking
    deepsort_cfg = get_config()
    deepsort_cfg.merge_from_file(opt.deepsort_cfg)
    deepsort_weights = opt.deepsort_weights

    # Load HRNet for detect keypoint
    update_config(cfg, opt)
    hrnet_model = eval(f"hrnet_models.{cfg.MODEL.NAME}.get_pose_net")(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        hrnet_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
    hrnet_model = torch.nn.DataParallel(hrnet_model, device_ids=cfg.GPUS)
    hrnet_model.to(device).eval()

    # Initial inference
    yolo_input = torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters()))
    yolo_model(yolo_input)
    hrnet_input = torch.zeros(1, 3, *cfg.MODEL.IMAGE_SIZE).to(device).type_as(next(hrnet_model.parameters()))
    hrnet_model(hrnet_input)

    for dataset in datasets:
        deepsort_model = DeepSort(deepsort_cfg.DEEPSORT.REID_CKPT,
                                  max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
                                  min_confidence=deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                                  max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                  max_age=deepsort_cfg.DEEPSORT.MAX_AGE, n_init=deepsort_cfg.DEEPSORT.N_INIT,
                                  nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
                                  use_cuda=True)
        action_dict = {}

        for img, bbox, keypoint in dataset:
            im0 = img[..., ::-1].copy()
            img, _, _ = letterbox(img, yolo_imgsz)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img = img / 255.
            if len(img.shape) == 3:
                img = img[None]

            t1 = time_sync()
            # YOLOv5 predict
            pred = yolo_model(img)[0]
            pred = non_max_suppression(pred, opt.yolo_conf_thr, opt.yolo_iou_thr, opt.yolo_target_classes,
                                       max_det=opt.yolo_max_det)
            for i, det in enumerate(pred):
                img_pose = im0.copy()
                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    clss = det[:, 5]
                    person_idx = clss == 0
                    xywhs = xyxy2xywh(det[:, 0:4])[person_idx]
                    confs = det[:, 4][person_idx]
                    clss = clss[person_idx]
                    outputs = deepsort_model.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    if len(outputs) > 0:
                        cropped_person_batch = []
                        person_centers = []
                        person_scales = []
                        person_ids = []

                        for j, (output, conf) in enumerate(zip(reversed(outputs), reversed(confs))):
                            xyxy = output[0: 4]
                            id = int(output[4])
                            cls = output[5]
                            c = int(cls)
                            label = f"{id} {names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(id, True))

                            if c == 0:
                                box = [(xyxy[0].item(), xyxy[1].item()), (xyxy[2].item(), xyxy[3].item())]
                                img_center = np.array([im0.shape[1] / 2, im0.shape[0] / 2])
                                center = np.array([(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2])
                                dist = np.linalg.norm(img_center - center)
                                if dist > 300:
                                    continue

                                person_crop = img_pose[int(box[0][1]): int(box[1][1]),
                                              int(box[0][0]): int(box[1][0])]
                                person_crop_lb, ratio, _ = letterbox(person_crop,
                                                                     (cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]),
                                                                     auto=False)
                                if 0 in ratio:
                                    continue
                                re_scale = [1 / ratio[0] * 1.25 * 1.15, 1 / ratio[1] * 1.25]

                                cropped_person_batch.append(transform(person_crop_lb).unsqueeze(0))
                                person_centers.append(center)
                                person_scales.append(re_scale)
                                person_ids.append(id)

                        if cropped_person_batch:
                            cropped_person_batch = torch.cat(cropped_person_batch)
                            kp_preds, kp_confs = get_pose_estimation_prediction_from_batch(hrnet_model,
                                                                                           cropped_person_batch,
                                                                                           person_centers,
                                                                                           person_scales,
                                                                                           cfg)
                            for kp_pred, kp_conf, person_id in zip(kp_preds, kp_confs, person_ids):
                                for kpt, kpc in zip(kp_pred, kp_conf):
                                    draw_pose(kpt, im0, kpc, opt.hrnet_vis_thr)
                                    tmp_kpt = torch.from_numpy(np.hstack((kpt, kpc))).view(-1).unsqueeze(0)
                                    tmp_kpt[:, 0::3] /= im0.shape[1]
                                    tmp_kpt[:, 1::3] /= im0.shape[0]
                                    if person_id not in action_dict:
                                        action_dict[person_id] = [tmp_kpt]
                                    else:
                                        action_dict[person_id].append(tmp_kpt)

                                action_input = torch.cat(action_dict[person_id])
                                print(action_input.shape)







            cv2.imshow("img", im0)
            cv2.waitKey(1)


def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5l_crowdhuman_v2.pt"
    # yolo_weights = "yolov5x.pt"
    parser.add_argument("--yolo_weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img", "--iyolo-mg-size", type=int, default=[640])
    parser.add_argument("--yolo-conf_thr", "--yolo-conf_thres", type=float, default=0.55)
    parser.add_argument("--yolo-iou-thr", "--yolo-iou-thres", type=float, default=0.3)
    parser.add_argument("--yolo-max-det", type=int, default=1000)
    parser.add_argument("--yolo-target-classes", default=None, nargs="+", type=int)
    parser.add_argument("--yolo-hide-labels", default=False, action="store_true")

    parser.add_argument("--deepsort-cfg", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--deepsort-weights", type=str, default="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

    hrnet_cfg = "inference-config.yaml"
    # hrnet_cfg = "hrnet_config.yaml"
    parser.add_argument("--hrnet-cfg", type=str, default=hrnet_cfg)
    parser.add_argument("--hrnet-opts", default=[])
    parser.add_argument("--hrnet-modelDir", default="")
    parser.add_argument("--hrnet-logDir", default="")
    parser.add_argument("--hrnet-dataDir", default="")
    parser.add_argument("--hrnet-vis-thr", type=float, default=0.6)

    parser.add_argument("--dataset-root", type=str, default="/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/")
    parser.add_argument("--video-num", type=int, default=45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)