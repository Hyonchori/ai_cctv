import argparse
import sys
import os
import collections
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_imshow, check_requirements, colorstr, is_ascii,\
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path, save_one_box
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox

from deep_sort_pytorch.utils.parser import get_config as get_deepsort_cfg
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.downloads import attempt_download

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "hrnet", "custom_lib"))
from hrnet.custom_lib import hrnet_models
from hrnet.custom_lib.config import cfg as hrnet_cfg
from hrnet.custom_lib.config import update_config
from hrnet.custom_lib.hrnet_utils.inference_utils import draw_pose, box_to_center_scale, \
    get_pose_estimation_prediction_from_batch, get_pose_estimation_prediction, transform, draw_keypoints

from stdet import StdetPredictor
from mmcv import Config as get_stdet_cfg
import mmcv
from custom_dataset_utils.ava_action_label import get_action_dict

from efficientnet.model import EfficientClassifier


def plot_action_label(img, actions, st, colors, verbose):
    location = (0 + st[0], 18 + verbose * 18 + st[1])
    diag0 = (location[0] + 20, location[1] - 14)
    diag1 = (location[0], location[1] + 2)
    cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
    if len(actions) > 0:
        for (label, score) in actions:
            text = f"{label}"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            textwidth = textsize[0]
            diag0 = (location[0] + textwidth, location[1] - 14)
            cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, 1)
            break


def plot_actions(img, bboxes, actions, ratio, colors, action_dict):
    for bbox, action in zip(bboxes, actions):
        bbox = bbox.cpu().numpy() / ratio[0]
        bbox = bbox.astype(np.int64)
        st, ed = tuple(bbox[:2]), tuple(bbox[2:])
        action = sorted(action, key=lambda x: x[1], reverse=True)
        action_pm = list(filter(lambda x: x[0] in action_dict["PERSON_MOVEMENT"], action))
        action_om = list(filter(lambda x: x[0] in action_dict["OBJECT_MANIPULATION"], action))
        action_pi = list(filter(lambda x: x[0] in action_dict["PERSON_INTERACTION"], action))
        plot_action_label(img, action_pm, st, colors, 0)
        plot_action_label(img, action_om, st, colors, 1)
        plot_action_label(img, action_pi, st, colors, 2)


@torch.no_grad()
def run(opt):
    # Load configs of YOLOv5
    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_max_det = opt.yolo_max_det
    yolo_target_clss = opt.yolo_target_clss
    yolo_save_crop = opt.yolo_save_crop

    # Load configs of DeepSORT
    deepsort_cfg = get_deepsort_cfg()
    deepsort_cfg.merge_from_file(opt.deepsort_cfg)
    deepsort_weights = opt.deepsort_weights

    # Load configs of STDet
    stdet_cfg = get_stdet_cfg.fromfile(opt.stdet_cfg)
    stdet_cfg.merge_from_dict(opt.stdet_cfg_options)
    stdet_weights = opt.stdet_weights
    stdet_interval = opt.stdet_interval
    stdet_action_score_thr = opt.stdet_action_score_thr
    stdet_label_map_path = opt.stdet_label_map_path
    stdet_img_norm_cfg = stdet_cfg["img_norm_cfg"]
    stdet_action_list_path = opt.stdet_action_list_path
    stdet_action_dict = get_action_dict(stdet_action_list_path)

    # Load configs of classifier
    clf_model_pt = opt.clf_model_pt
    clf_imgsz = opt.clf_imgsz
    clf_label_map_path = opt.clf_label_map_path
    clf_label_map = [x.replace("\n", "") for x in open(clf_label_map_path).readlines()]
    clf_thr = opt.clf_thr

    # Load other configs for inference
    source = opt.source
    device = opt.device
    dir_path = opt.dir_path
    run_name = opt.run_name
    is_video_frames = opt.is_video_frames
    save_vid = opt.save_vid
    model_usage = opt.model_usage
    show_vid = list(np.array(model_usage) & np.array(opt.show_vid))
    face_mosaic = opt.face_mosaic
    show_cls = opt.show_cls

    # Directories result will be saved
    save_dir = increment_path(Path(dir_path) / run_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    device = torch.device("cpu")

    # Load YOLOv5 model
    yolo_model = attempt_load(yolo_weights, map_location=device)
    stride = int(yolo_model.stride.max())
    names = yolo_model.module.names if hasattr(yolo_model, "module") else yolo_model.names
    yolo_imgsz = check_img_size(yolo_imgsz, s=stride)
    ascii = is_ascii(names)

    if model_usage[3]:
        # Load configs of HRNet
        update_config(hrnet_cfg, opt)
        hrnet_vis_thr = opt.hrnet_vis_thr

        # Load HRNet model
        hrnet_model = eval(f"hrnet_models.{hrnet_cfg.MODEL.NAME}.get_pose_net")(hrnet_cfg, is_train=False)
        if hrnet_cfg.TEST.MODEL_FILE:
            print('=> loading model from {}'.format(hrnet_cfg.TEST.MODEL_FILE))
            hrnet_model.load_state_dict(torch.load(hrnet_cfg.TEST.MODEL_FILE), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')
        hrnet_model = torch.nn.DataParallel(hrnet_model, device_ids=hrnet_cfg.GPUS)
        hrnet_model.to(device).eval()
        hrnet_imgsz = hrnet_cfg.MODEL.IMAGE_SIZE

    # Load STDet model
    stdet_model = StdetPredictor(
        config=stdet_cfg,
        checkpoint=stdet_weights,
        device=device,
        score_thr=stdet_action_score_thr,
        label_map_path=stdet_label_map_path
    )

    # Load Classifier
    clf_model = torch.load(clf_model_pt)
    clf_model = EfficientClassifier(model_version="efficientnet-b1", num_classes=2).cuda().eval()
    clf_model.load_state_dict(torch.load(clf_model_pt))
    clf_pp = transforms.Compose([
        #transforms.Resize(128),
        #transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # DataLoader
    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    ) or source.startswith("/dev/video")
    if webcam:
        dataset = LoadStreams(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Load DeepSORT model
    deepsort_model_list = [DeepSort(deepsort_cfg.DEEPSORT.REID_CKPT,
                                    max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
                                    min_confidence=deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                                    max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=deepsort_cfg.DEEPSORT.MAX_AGE,
                                    n_init=deepsort_cfg.DEEPSORT.N_INIT,
                                    nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True) for _ in range(bs)]

    # Run inference
    action_input = {
        "img": [torch.zeros(1, 3, 8, 256, 256).to(device).type_as(next(stdet_model.model.parameters()))],
        "img_metas": [[{"img_shape": (256, 256)}]],
        "proposals": [[torch.tensor([[10, 10, 20, 20]], device=device).type_as(next(stdet_model.model.parameters()))]],
        "return_loss": False
    }
    if device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters())))
        #hrnet_model(torch.zeros(1, 3, *hrnet_imgsz).to(device).type_as(next(hrnet_model.parameters())))
        clf_model(torch.zeros(1, 3, *clf_imgsz).to(device).type_as(next(clf_model.parameters())))
        stdet_model.model(**action_input)

    stdet_input_imgs = [collections.deque([], maxlen=8) for _ in range(bs)]
    tmp_interval = 0
    for path, img, im0s, vid_cap in dataset:
        print("\n---")
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.

        if len(img.shape) == 3:
            img = img[None]

        t1 = time_sync()

        # YOLO inference
        if model_usage[0]:
            yolo_pred = yolo_model(img)[0]
            yolo_pred = non_max_suppression(yolo_pred, yolo_conf_thr, yolo_iou_thr, yolo_target_clss, max_det=yolo_max_det)
        else:
            yolo_pred = []

        for i, det in enumerate(yolo_pred):
            if webcam:
                p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / "video") if is_video_frames else str(save_dir / p.name)
            s += "%g%g" % img.shape[2:]
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if face_mosaic:
                    for box in det:
                        xyxy = box[:4]
                        if box[-1] == 1 or box[-1] == 2:
                            face = im0[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                            face = cv2.resize(face, dsize=(10, 10))
                            face = cv2.resize(face, (int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1])),
                                              interpolation=cv2.INTER_AREA)
                            im0[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])] = face

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                if model_usage[1]:
                    clss = det[:, 5]
                    person_idx = clss == 0
                    xywhs = xyxy2xywh(det[:, :4])[person_idx]
                    confs = det[:, 4][person_idx]
                    clss = clss[person_idx]
                    deepsort_pred = deepsort_model_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                box_show = show_vid[0] or show_vid[1] or show_vid[2]
                if show_vid[0] and not show_vid[1]:
                    box_iter = det
                elif show_vid[1]:
                    box_iter = deepsort_pred
                else:
                    box_iter = det

                if len(box_iter) > 0:
                    tmp_proposals = []
                    if model_usage[4] and tmp_interval % stdet_interval == 0:
                        stdet_input_size = mmcv.rescale_size((im0.shape[1], im0.shape[0]), (256, np.inf))
                        if "to_rgb" not in stdet_img_norm_cfg and "to_bgr" in stdet_img_norm_cfg:
                            to_bgr = stdet_img_norm_cfg.pop("to_bgr")
                            stdet_img_norm_cfg["to_rgb"] = to_bgr
                        stdet_img_norm_cfg["mean"] = np.array(stdet_img_norm_cfg["mean"])
                        stdet_img_norm_cfg["std"] = np.array(stdet_img_norm_cfg["std"])

                        stdet_input_img = mmcv.imresize(im0, stdet_input_size).astype(np.float32)
                        _ = mmcv.imnormalize_(stdet_input_img, **stdet_img_norm_cfg)
                        stdet_input_imgs[i].append(stdet_input_img)
                        ratio = (stdet_input_size[0] / im0.shape[1], stdet_input_size[1] / im0.shape[0])
                        tmp_interval = 0
                    tmp_interval += 1

                    for output in box_iter:
                        xyxy = output[: 4]
                        cls = int(output[5]) if show_vid[1] else int(output[-1])
                        label = f"{names[cls]}"

                        if model_usage[2]:
                            if cls != 0:
                                continue
                            tmp = im0s[i].copy() if webcam else im0s.copy()
                            clf_input_img = tmp[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                            if 0 in clf_input_img.shape:
                                continue
                            clf_input_img, _, _ = letterbox(clf_input_img, clf_imgsz, auto=False)
                            clf_input = torch.from_numpy(clf_input_img.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
                            clf_input /= 255.
                            clf_pred = clf_model(clf_input)
                            print(clf_pred)
                            clf_idx = torch.argmax(clf_pred, dim=1)[0]
                            clf_val = torch.max(clf_pred, dim=1)[0]
                            if clf_idx == 0:
                                if clf_val >= clf_thr + 0.3 and show_vid[2]:
                                    label = f"{clf_label_map[clf_idx]} {clf_val.item():.2f}"
                                    #label = f"{clf_label_map[clf_idx]}"
                            elif clf_idx == 1:
                                if clf_val >= clf_thr and show_vid[2]:
                                    label = f"{clf_label_map[clf_idx]} {clf_val.item():.2f}"
                                    #label = f"{clf_label_map[clf_idx]}"
                        if box_show and cls == show_cls:
                            if not model_usage[2]:
                                id = int(output[4]) if show_vid[1] else int(output[-1])
                            else:
                                if "person" in label:
                                    id = 0
                                elif "civil" in label:
                                    id = 1
                                elif "military" in label:
                                    id = 2
                            annotator.box_label(xyxy, label, color=colors(id, True))

                        if model_usage[4] and cls == 0:
                            if isinstance(xyxy, np.ndarray):
                                proposal = torch.from_numpy(xyxy * ratio[0]).unsqueeze(0).to(device)
                            else:
                                proposal = xyxy * ratio[0]
                                proposal = proposal.unsqueeze(0).to(device)
                            tmp_proposals.append(proposal)

                    if len(tmp_proposals) > 0:
                        tmp_proposals = [[torch.cat(tmp_proposals).float()]]

                        if len(stdet_input_imgs[i]) == 8:
                            imgs = np.stack(stdet_input_imgs[i]).transpose(3, 0, 1, 2)
                            imgs = [torch.from_numpy(imgs).unsqueeze(0).to(device)]

                            img_meta = [[{"img_shape": stdet_input_img.shape[:2]}]]
                            return_loss = False

                            stdet_input = {
                                "img": imgs,
                                "img_metas": img_meta,
                                "proposals": tmp_proposals,
                                "return_loss": return_loss
                            }
                            stdet_pred = stdet_model.model(**stdet_input)[0]
                            stdet_result = []
                            for _ in range(tmp_proposals[0][0].shape[0]):
                                stdet_result.append([])
                            for class_id in range(len(stdet_pred)):
                                if class_id + 1 not in stdet_model.label_map:
                                    continue
                                for bbox_id in range(tmp_proposals[0][0].shape[0]):
                                    if len(stdet_pred[class_id]) != tmp_proposals[0][0].shape[0]:
                                        continue
                                    if stdet_pred[class_id][bbox_id, 4] > stdet_model.score_thr:
                                        stdet_result[bbox_id].append((stdet_model.label_map[class_id + 1],
                                                                      stdet_pred[class_id][bbox_id, 4]))
                            if show_vid[4]:
                                plot_actions(im0, tmp_proposals[0][0], stdet_result, ratio, colors, stdet_action_dict)
            else:
                if model_usage[1]:
                    deepsort_model_list[i].increment_ages()

            t2 = time_sync()
            if isinstance(vid_cap, list):
                pass
                #print(vid_cap[i].get(cv2.CAP_PROP_FRAME_COUNT))
                #print(vid_cap[i].get(cv2.CAP_PROP_POS_FRAMES))
            elif vid_cap is not None:
                pass
                #print(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #print(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                pass
            print(f"elapsed time: {t2 - t1:.4f}")
            if any(show_vid):
                cv2.imshow("img", im0)
                cv2.waitKey(1)

            if save_vid:
                if dataset.mode == "image" and not is_video_frames:
                    cv2.imwrite(save_path, im0)
                else:
                    cap = vid_cap[i] if isinstance(vid_cap, list) else vid_cap
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if cap:
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                        save_path += ".mp4"
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)


def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5/yolov5l_crowdhuman_v4.pt"
    #yolo_weights = "yolov5x.pt"
    parser.add_argument("--yolo_weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img", "--yolo-img-size", type=int, default=[640])
    parser.add_argument("--yolo-conf_thr", "--yolo-conf_thres", type=float, default=0.5)
    parser.add_argument("--yolo-iou-thr", "--yolo-iou-thres", type=float, default=0.6)
    parser.add_argument("--yolo-max-det", type=int, default=1000)
    parser.add_argument("--yolo-target-clss", default=[0, 1], nargs="+", type=int)
    parser.add_argument("--yolo-save-crop", default=False, action="store_true")

    parser.add_argument("--deepsort-cfg", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--deepsort-weights", type=str, default="weights/deep_sort/deep/checkpoint/ckpt.t7")

    hrnet_cfg = "weights/hrnet/hrnet-config.yaml"
    parser.add_argument("--hrnet-cfg", type=str, default=hrnet_cfg)
    parser.add_argument("--hrnet-opts", default=[])
    parser.add_argument("--hrnet-modelDir", default="")
    parser.add_argument("--hrnet-logDir", default="")
    parser.add_argument("--hrnet-dataDir", default="")
    parser.add_argument("--hrnet-vis-thr", type=float, default=0.6)

    stdet_cfg = "mmaction2/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py"
    stdet_weights = "weights/stdet/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305-c6225546.pth"
    parser.add_argument("--stdet-cfg", default=stdet_cfg)
    parser.add_argument("--stdet-weights", default=stdet_weights)
    parser.add_argument("--stdet-interval", type=int, default=1)
    parser.add_argument("--stdet-action-score-thr", type=float, default=0.4)
    parser.add_argument("--stdet-action-list-path", default="weights/stdet/ava_action_list_v2.2.pbtxt")
    parser.add_argument("--stdet-label-map-path", default="mmaction2/tools/data/ava/label_map.txt")
    parser.add_argument("--stdet-cfg-options", default={})

    clf_model_pt = "weights/classifier/military_civil_clf_f.pt"
    #clf_model_pt = "/home/daton/PycharmProjects/pythonProject/sam/example/weights/effb1-96.pt"
    parser.add_argument("--clf-model_pt", type=str, default=clf_model_pt)
    parser.add_argument("--clf-imgsz", type=int, default=[128])
    parser.add_argument("--clf-label-map-path", default="weights/classifier/military_civil_label_map.txt")
    parser.add_argument("--clf-thr", type=float, default=0.6)

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/train/MOT17-04-DPM/img1"
    source = "0"
    source = "/media/daton/SAMSUNG1/지하철 역사 내 CCTV 이상행동 영상/Training/에스컬레이터 전도/에스컬레이터 전도_2/3118061"

    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--dir_path", default="runs/inference")
    parser.add_argument("--run_name", default="exp")
    parser.add_argument("--is_video_frames", type=bool, default=True)
    parser.add_argument("--save-vid", type=bool, default=True)
    show_vid = [1, 1, 1, 1, 1]  # idx 0=yolo, 1=deepsort, 2=classifier, 3=hrnet, 4=stdet
    parser.add_argument("--show-vid", type=list, default=show_vid)
    parser.add_argument("--face_mosaic", type=bool, default=False)
    model_usage = [1, 0, 0, 0, 1]  # idx 0=yolo, 1=deepsort, 2=classifier, 3=hrnet, 4=stdet
    parser.add_argument("--model-usage", type=list, default=model_usage)
    parser.add_argument("--show_cls", type=int, default=0)

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1  # expand
    opt.clf_imgsz *= 2 if len(opt.clf_imgsz) == 1 else 1
    return opt


def main(opt):
    print(colorstr('Inference: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(requirements="yolov5/requirements.txt", exclude=("tensorboard", "thop"))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)