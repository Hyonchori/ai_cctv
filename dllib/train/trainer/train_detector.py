
import os
import argparse
import copy
import time
import csv
from tqdm import tqdm

import torch
import torch.optim as optim
import albumentations as A

from dllib.train.losses import ComputeDetectionLoss
from dllib.train.lr_schedulers import CosineAnnealingWarmUpRestarts
from dllib.data.for_train.coco_dataset import get_coco2017dataloader
from dllib.models.detector import BuildDetector


def main(opt):
    model = BuildDetector(backbone_cfg=opt.backbone_cfg,
                          neck_cfg=opt.neck_cfg,
                          detector_head_cfg=opt.head_cfg,
                          info=True).cuda()
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
    device = next(model.parameters()).device

    train_transform = A.Compose([
        A.ColorJitter(),
        A.HueSaturationValue(),
        A.RandomBrightnessContrast(),
        A.Cutout(max_h_size=32, max_w_size=32)
    ])
    train_dataloader, valid_dataloader = get_coco2017dataloader(img_size=opt.img_size,
                                                                mode="detection",
                                                                train_batch=opt.batch_size,
                                                                train_transform=train_transform)
    compute_loss = ComputeDetectionLoss(model)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    lr_sch = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.001, T_up=3, gamma=0.7)

    loss_weight = [1, 0.5, 1]  # iou, conf, cls

    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_name = opt.name
    wts_save_dir = os.path.join(save_dir, model_name + ".pt")
    log_save_dir = os.path.join(save_dir, model_name + "_log.csv")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.
    save_interval = opt.save_interval

    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    for e in range(start_epoch, end_epoch+1):
        print("\n--- {}".format(e))
        time.sleep(0.5)
        train_loss = train(model, optimizer, e, train_dataloader, compute_loss, loss_weight, device)
        time.sleep(0.5)
        print(train_loss)

        valid_loss = evaluate(model, e, valid_dataloader, compute_loss, device)
        time.sleep(0.5)
        print(valid_loss)

        if os.path.isfile(log_save_dir):
            with open(log_save_dir, "r") as f:
                reader = csv.reader(f)
                logs = list(reader)
                logs.append(train_loss + valid_loss + [optimizer.param_groups[0]["lr"]])
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                for log in logs:
                    writer.writerow(log)
        else:
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                writer.writerow(train_loss + valid_loss + [optimizer.param_groups[0]["lr"]])

        for k, (l, w) in enumerate(zip(valid_loss, loss_weight)):
            valid_loss[k] = valid_loss[k] * w
        loss = sum(valid_loss)
        if loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())

        lr_sch.step()
        if e % save_interval == 0:
            torch.save(best_model_wts, wts_save_dir)


def train(model, optimizer, epoch, dataloader, compute_loss, loss_weight, device):
    model.train()
    train_loss = [0., 0., 0.]  # [iou, conf, cls]
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, bboxes0, bbox_b, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        for i, bbox in enumerate(bbox_b):
            bbox_b[i] = bbox_b[i].to(device)
        optimizer.zero_grad()
        pred = model(img_b.float(), epoch)

        for i in range(len(pred)):
            bs, _, _, _, no = pred[i].shape
            pred[i] = pred[i].view(bs, -1, no)
        pred = torch.cat(pred, dim=1)

        loss = compute_loss(pred, bbox_b)
        for j, _ in enumerate(train_loss):
            train_loss[j] += loss[j].item()

        for k, (l, w) in enumerate(zip(loss, loss_weight)):
            loss[k] = loss[k] * w
        loss = loss.sum()
        loss.backward()
        optimizer.step()

    for j, _ in enumerate(train_loss):
        train_loss[j] /= len(pbar)
    return train_loss


@torch.no_grad()
def evaluate(model, epoch, dataloader, compute_loss, device):
    model.eval()
    train_loss = [0., 0., 0.]  # [iou, conf, cls]
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, bboxes0, bbox_b, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        for i, bbox in enumerate(bbox_b):
            bbox_b[i] = bbox_b[i].to(device)
        pred = model(img_b.float(), epoch)

        for i in range(len(pred)):
            bs, _, _, _, no = pred[i].shape
            pred[i] = pred[i].view(bs, -1, no)
        pred = torch.cat(pred, dim=1)

        loss = compute_loss(pred, bbox_b)
        for j, _ in enumerate(train_loss):
            train_loss[j] += loss[j].item()

    for j, _ in enumerate(train_loss):
        train_loss[j] /= len(pbar)
    return train_loss


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_backbone_m.yaml")
    parser.add_argument("--neck_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_neck_m.yaml")
    parser.add_argument("--head_cfg", type=str, help="backbone.yaml path",
                        default="../../models/cfgs/base_detection_head2.yaml")
    parser.add_argument("--weights", type=str, help="initial weights path",
                        default="../../weights/base_detector.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--save_dir", type=str, default="../../weights")
    parser.add_argument("--name", type=str, default="base_detector")
    parser.add_argument("--save_interval", type=int, default=10)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
