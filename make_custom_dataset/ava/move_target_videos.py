import shutil
import os


if __name__ == "__main__":
    target_video_file = "target_videos.txt"
    with open(target_video_file) as f:
        target_videos = [x.replace("\n", "") for x in f.readlines()]
        print(target_videos)
        print(len(target_videos))

    input_root = "/media/daton/Data/dataset/ava/videos_15min"
    output_root = "/media/daton/Data/dataset/ava/target_videos_15min"

    vids = os.listdir(input_root)
    for vid in vids:
        vid_name = vid.split(".")[0]
        if vid_name in target_videos:
            vid_path = os.path.join(input_root, vid)
            out_path = os.path.join(output_root, vid)
            shutil.copy(vid_path, out_path)