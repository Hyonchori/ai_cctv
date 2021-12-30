import os
import shutil


if __name__ == "__main__":
    target_videos_file = "complete_videos.txt"
    with open(target_videos_file) as f:
        target_videos = [x.replace("\n", "").split("/")[-1].split(".")[0] for x in f.readlines()]
        print(target_videos)
        print(len(target_videos))

    target_root = "/media/daton/Data/dataset/ava/rawframes"
    candi_videos = [x for x in os.listdir(target_root) if x in target_videos]
    cnt = 0
    for i, candi in enumerate(candi_videos):
        print(f"\n--- {candi}   {i + 1} / {len(candi_videos)}")
        tmp_path = os.path.join(target_root, candi)
        shutil.make_archive(tmp_path, "zip", tmp_path)
        shutil.rmtree(tmp_path)
        cnt += 1
    print(cnt)