
import os
import argparse
import copy
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm

from dllib.data.for_train.military_civil_dataset import get_mc_train_dataloader, \
    get_mc_valid_dataloader
from dllib.train.losses import FocalLoss
from dllib.train.lr_schedulers import CosineAnnealingWarmUpRestarts
from ai_cctv.efficientnet.model import EfficientClassifier


def main(opt):
    model = EfficientClassifier(model_version="efficientnet-b1").cuda()
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
            print("Model is initialized with existing weights!")
        else:
            print("Model is initialized!")
    device = next(model.parameters()).device

    train_transform = A.Compose([
        A.RandomBrightnessContrast(),
        #A.HorizontalFlip(),
        A.Rotate(limit=20),
        A.RandomContrast(),
        A.RandomGamma(),
        A.Cutout(num_holes=4, max_w_size=12, max_h_size=12)
    ])
    train_dataloader = get_mc_train_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=train_transform)
    valid_dataloader = get_mc_valid_dataloader(img_size=opt.img_size,
                                               batch_size=opt.batch_size,
                                               transform=None)

    #loss_fn = nn.MSELoss()
    #loss_fn = nn.BCELoss()
    loss_fn = FocalLoss(nn.BCELoss(weight=torch.tensor([1., 1.3]).to(device)))
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.003, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.003)
    #lr_sch = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.0001, T_up=3, gamma=0.95)

    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_name = opt.name
    wts_save_dir = os.path.join(save_dir, model_name + ".pt")
    log_save_dir = os.path.join(save_dir, model_name + "_log.csv")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.
    best_acc = 0.
    save_interval = opt.save_interval

    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    for e in range(start_epoch, end_epoch + 1):
        print(f"\n--- Epoch: {e}")
        time.sleep(0.5)
        train_loss = train(model, optimizer, e, train_dataloader, loss_fn, device)
        time.sleep(0.5)
        print("")
        print(train_loss)

        valid_loss = evaluate(model, e, valid_dataloader, loss_fn, device)
        time.sleep(0.5)
        print("")
        print(valid_loss)

        if os.path.isfile(log_save_dir):
            with open(log_save_dir, "r") as f:
                reader = csv.reader(f)
                logs = list(reader)
                logs.append([x for x in train_loss] + [x for x in valid_loss] + [optimizer.param_groups[0]["lr"]])
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                for log in logs:
                    writer.writerow(log)
        else:
            with open(log_save_dir, "w") as f:
                writer = csv.writer(f)
                writer.writerow([x for x in train_loss] + [x for x  in valid_loss] + [optimizer.param_groups[0]["lr"]])

        if valid_loss[1] > best_acc:
            best_acc = valid_loss[1]
            best_model_wts = copy.deepcopy(model.state_dict())

        #lr_sch.step()
        if e % save_interval == 0:
            torch.save(best_model_wts, wts_save_dir)


def train(model, optimizer, epoch, dataloader, loss_fn, device):
    model.train()
    train_loss = 0.
    train_acc = 0.
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, img_cls, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        img_cls = torch.from_numpy(img_cls).to(device).float()

        optimizer.zero_grad()
        pred = model(img_b.float(), epoch)
        loss = loss_fn(pred, img_cls)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            gt_cls = torch.argmax(img_cls, dim=1)
            pred_cls = torch.argmax(pred, dim=1)
            correct = gt_cls == pred_cls
            accuracy = torch.sum(torch.ones_like(gt_cls)[correct]) / gt_cls.shape[0]

        train_loss += loss.item()
        train_acc += accuracy.item()

    train_loss /= len(pbar)
    train_acc /= len(pbar)
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, epoch, dataloader, loss_fn, device):
    model.eval()
    valid_loss = 0.
    valid_acc = 0.
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    for i, (img0, img_b, img_cls, img_name) in pbar:
        img_b = img_b.to(device) / 255.
        img_cls = torch.from_numpy(img_cls).to(device).float()

        pred = model(img_b.float(), epoch)
        loss = loss_fn(pred, img_cls)

        gt_cls = torch.argmax(img_cls, dim=1)
        pred_cls = torch.argmax(pred, dim=1)
        correct = gt_cls == pred_cls
        accuracy = torch.sum(torch.ones_like(gt_cls)[correct]) / gt_cls.shape[0]
        valid_loss += loss.item()
        valid_acc += accuracy.item()

    valid_loss /= len(pbar)
    valid_acc /= len(pbar)
    return valid_loss, valid_acc


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="../weights/classifier/military_civil_clf_b.pt")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--save_dir", type=str, default="../weights/classifier")
    parser.add_argument("--name", type=str, default="military_civil_clf_b")
    parser.add_argument("--save_interval", type=int, default=10)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
