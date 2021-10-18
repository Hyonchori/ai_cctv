import argparse
import sys
import os
import time
import collections
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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

from deep_sort_pytorch.utils.parser import get_config as get_deepsort_cfg
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.downloads import attempt_download

from stdet import StdetPredictor
from mmcv import Config as get_stdet_cfg
import mmcv
from custom_dataset.ava_action_label import action_dict


def plot_action_label(img, actions, st, colors, verbose):
    location = (0 + st[0], 18 + verbose * 18 + st[1])
    diag0 = (location[0] + 20, location[1] - 14)
    diag1 = (location[0], location[1] + 2)
    cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
    if len(actions) > 0:
        for (label, score) in actions:
            text = f"{label}: {score:.2f}"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            textwidth = textsize[0]
            diag0 = (location[0] + textwidth, location[1] - 14)
            cv2.rectangle(img, diag0, diag1, colors(verbose + 110, True), -1)
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, 1)
            break


def plot_actions(img, bboxes, actions, ratio, colors):
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

    tmp_action_input = {
        "img": [torch.zeros(1, 3, 8, 256, 256).to(device).type_as(next(stdet_model.model.parameters()))],
        "img_metas": [[{"img_shape": (256, 256)}]],
        "proposals": [[torch.tensor([[10, 10, 20, 20]], device=device).type_as(next(stdet_model.model.parameters()))]],
        "return_loss": False
    }
    img_norm_cfg = stdet_cfg["img_norm_cfg"]

    # Run inference
    if device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.parameters())))
        stdet_model.model(**tmp_action_input)

    t0 = time.time()
    action_input_imgs = [collections.deque([], maxlen=8) for _ in range(bs)]
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

            p = Path(p)
            save_path = str(save_dir / "video") if "MOT17" in str(p) else str(save_dir / p.name)
            s += "%gx%g" % img.shape[2:]
            imc = im0.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)



            if len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                clss = det[:, 5]
                person_idx = clss == 0
                xywhs = xyxy2xywh(det[:, 0:4])[person_idx]
                confs = det[:, 4][person_idx]
                clss = clss[person_idx]
                outputs = deepsort_model_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                if len(outputs) > 0:

                    stdet_input_size = mmcv.rescale_size((im0.shape[1], im0.shape[0]), (256, np.inf))
                    if "to_rgb" not in img_norm_cfg and "to_bgr" in img_norm_cfg:
                        to_bgr = img_norm_cfg.pop("to_bgr")
                        img_norm_cfg["to_rgb"] = to_bgr
                    img_norm_cfg["mean"] = np.array(img_norm_cfg["mean"])
                    img_norm_cfg["std"] = np.array(img_norm_cfg["std"])

                    processed_frame = mmcv.imresize(im0, stdet_input_size).astype(np.float32)
                    _ = mmcv.imnormalize_(processed_frame, **img_norm_cfg)
                    action_input_imgs[i].append(processed_frame)


                    cv2.imshow("img", action_input_imgs[i][-1])

                    tmp_proposals = []
                    ratio = (stdet_input_size[0] / im0.shape[1], stdet_input_size[1] / im0.shape[0])
                    for j, (output, conf) in enumerate(zip(reversed(outputs), reversed(confs))):
                        xyxy = output[0: 4]
                        id = int(output[4])
                        cls = output[5]
                        c = int(cls)
                        label = f"{id} {names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(id, True))

                        proposal = torch.from_numpy(xyxy * ratio[0]).unsqueeze(0).to(device)
                        tmp_proposals.append(proposal)

                    if len(tmp_proposals) > 0:
                        tmp_proposals = [[torch.cat(tmp_proposals).float()]]

                        if len(action_input_imgs[i]) == 8:
                            imgs = np.stack(action_input_imgs[i]).transpose(3, 0, 1, 2)
                            imgs = [torch.from_numpy(imgs).unsqueeze(0).to(device)]


                            img_meta = [[{"img_shape": processed_frame.shape[:2]}]]
                            return_loss = False

                            action_input = {
                                "img": imgs,
                                "img_metas": img_meta,
                                "proposals": tmp_proposals,
                                "return_loss": return_loss
                            }
                            result = stdet_model.model(**action_input)[0]
                            preds = []
                            for _ in range(tmp_proposals[0][0].shape[0]):
                                preds.append([])
                            for class_id in range(len(result)):
                                if class_id + 1 not in stdet_model.label_map:
                                    continue
                                for bbox_id in range(tmp_proposals[0][0].shape[0]):
                                    if len(result[class_id]) != tmp_proposals[0][0].shape[0]:
                                        continue
                                    if result[class_id][bbox_id, 4] > stdet_model.score_thr:
                                        preds[bbox_id].append((stdet_model.label_map[class_id + 1],
                                                               result[class_id][bbox_id, 4]))
                            plot_actions(im0, tmp_proposals[0][0], preds, ratio, colors)

            else:
                deepsort_model_list[i].increment_ages()


            t3 = time_sync()
            print(f"elapsed time: {t3 - t1:.4f}")
            #cv2.imshow(str(p), im0)
            cv2.imshow("img", im0)
            cv2.waitKey(1)

            if opt.save_vid:
                if dataset.mode == "imagea":
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
                            fps, w, h = 15, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        print(fps, w, h)
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)





def parse_opt():
    parser = argparse.ArgumentParser()

    yolo_weights = "weights/yolov5/yolov5l_crowdhuman_v3.pt"
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
    parser.add_argument("--deepsort-weights", type=str, default="weights/deep_sort/ckpt.t7")

    stdet_cfg = "mmaction2/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py"
    stdet_weights = "weights/stdet/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-b987b516.pth"
    parser.add_argument("--stdet-cfg", default=stdet_cfg)
    parser.add_argument("--stdet-weights", default=stdet_weights)
    parser.add_argument("--stdet-action-score-thr", type=float, default=0.4)
    parser.add_argument("--stdet-label-map-path", default="../mmaction2/tools/data/ava/label_map.txt")
    parser.add_argument("--stdet-cfg-options", default={})

    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    #source = "source.txt"
    source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/train/MOT17-02-DPM/img1"
    #source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/test/MOT17-03-DPM/img1"
    #source = "/media/daton/D6A88B27A88B0569/dataset/사람동작 영상/이미지/image_action_45/image_45-1/45-1/45-1_001-C01"
    #source = "https://www.youtube.com/watch?v=-gSOi6diYzI"
    #source = "https://www.youtube.com/watch?v=gwavBeK4H1Q"
    #source = "/home/daton/Downloads/videos/bandicam 2021-09-24 05-22-34-452.mp4"
    source = "https://www.youtube.com/watch?v=rnGlZYcn0Z4"
    source = "0"
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