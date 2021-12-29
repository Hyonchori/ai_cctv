import os

import cv2
import pandas


def get_target_videos(annot, vids, target_actions):
    target_vids = []
    for vid_name in vids:
        vid_idx = (annot[:, 0] == vid_name)
        vid = annot[vid_idx]
        is_target = False
        for tmp in vid:
            tmp_action = tmp[-2]
            if tmp_action in target_actions:
                is_target = True
                break
        if is_target:
            target_vids.append(vid_name)
    return target_vids


if __name__ == "__main__":
    annot_root = "/media/daton/Data/dataset/ava/annotations"
    train_file = os.path.join(annot_root, "ava_train_v2.1.csv")
    valid_file = os.path.join(annot_root, "ava_val_v2.1.csv")

    video_root = "/media/daton/Data/dataset/ava/videos_15min"
    video_name = "-5KQ66BBWC4.mkv"

    train_annot = pandas.read_csv(train_file).to_numpy()
    print(train_annot)
    valid_annot = pandas.read_csv(valid_file).to_numpy()
    print(valid_annot)

    train_vids = list(set(train_annot[:, 0]))
    print(len(train_vids))
    valid_vids = list(set(valid_annot[:, 0]))
    print(len(valid_vids))

    target_actions = [5, 8]  # 5: fall down, 8: lie/sleep

    train_target_vids = get_target_videos(train_annot, train_vids, target_actions)
    valid_target_vids = get_target_videos(valid_annot, valid_vids, target_actions)
    total_target_vids = train_target_vids + valid_target_vids

    '''print(len(train_target_vids))
    with open("target_train_videos.txt", "w") as f:
        f.write("\n".join(train_target_vids))

    print(len(valid_target_vids))
    with open("target_valid_videos.txt", "w") as f:
        f.write("\n".join(valid_target_vids))'''

    print(len(total_target_vids))
    with open("target_videos.txt", "w") as f:
        f.write("\n".join(total_target_vids))