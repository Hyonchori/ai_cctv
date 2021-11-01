import wget
import os

trainval_videos_file = "../ava_trainval_videos"
test_videos_file = "../ava_test_videos"

def get_videos(file):
    with open(file) as f:
        data = f.readlines()
        data = [x.replace("\n", "") for x in data]
    return data

trainval_videos = get_videos(trainval_videos_file)
test_videos = get_videos(test_videos_file)


def download_videos(videos, dir):
    trainval = "trainval" if "train" in dir else "test"
    for i in range(len(videos)):
        print(f"\n--- {trainval} {i+1}/{len(videos)}")
        print(videos[i])
        if videos[i] in os.listdir(dir):
            continue
        url = f"https://s3.amazonaws.com/ava-dataset/{trainval}/{videos[i]}"
        wget.download(url, out=dir)


train_path = "/media/daton/D6A88B27A88B0569/dataset/ava/train/videos"
ref_path = "/media/daton/D6A88B27A88B0569/dataset/ava/train/videos_15min"
files1 = trainval_videos
files2 = os.listdir(ref_path)

inter = set(files1) & set(files2)
excl = set(files1) - set(files2)
print(len(inter))
print(len(excl))
print(len(files1))

test_path = "/media/daton/D6A88B27A88B0569/dataset/ava/test"

download_videos(list(excl), train_path)
#download_videos(test_videos, test_path)