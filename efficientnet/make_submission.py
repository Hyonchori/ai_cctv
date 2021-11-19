import os
import argparse

import pandas as pd
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from ai_cctv.efficientnet.model import EfficientClassifier
from dllib.utils.img_utils import letterbox


def main(opt):
    root = "/home/daton/Downloads/dataset"
    submission_path = os.path.join(root, "sample_submission.csv")
    submission = pd.read_csv(submission_path)

    test_dir = os.path.join(root, "test")
    test_imgs = [x for x in sorted(os.listdir(test_dir)) if x.endswith("png")]

    model = EfficientClassifier(model_version="efficientnet-b1", num_classes=10).cuda().eval()
    if opt.weights is not None:
        if os.path.isfile(opt.weights):
            wts = torch.load(opt.weights)
            model.load_state_dict(wts)
            print("Model is initialized with existing weights!")
        else:
            print("Model is initialized!")
    device = next(model.parameters()).device

    preds = []

    for img_name in tqdm(test_imgs):
        img_path = os.path.join(test_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        img_resize, _, _ = letterbox(img, opt.img_size, auto=False)
        img_torch = img_resize[..., ::-1].transpose(2, 0 ,1)
        img_torch = np.ascontiguousarray(img_torch)
        img_torch = torch.from_numpy(img_torch).unsqueeze(0)

        img_torch = img_torch.to(device) / 255.
        pred = model(img_torch.float())
        prediction = torch.argmax(pred, dim=1)
        preds.append(prediction.cpu().item())

    submission["label"] = preds
    print(submission)
    submission.to_csv('submission_v3.csv', index=False)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="../weights/classifier/mnist5.pt")
    parser.add_argument("--img_size", type=int, default=196)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt(True)
    main(opt)