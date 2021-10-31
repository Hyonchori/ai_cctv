
import os
import argparse
import copy
import time
import csv

import torch
import torch.nn as nn
import albumentations as A
from tqdm import tqdm

from dllib.data.for_train.military_civil_dataset import get_mc_train_dataloader, \
    get_mc_valid_dataloader
from ai_cctv.efficientnet.model import EfficientClassifier


def main(opt):
    model = EfficientClassifier().cuda().eval()
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
    loss_fn = nn.BCELoss()

    valid_loss = evaluate(model, 0, valid_dataloader, loss_fn, device)
    time.sleep(0.5)
    print("")
    print(valid_loss)


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
                        default="../weights/classifier/military_civil_clf11.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)
