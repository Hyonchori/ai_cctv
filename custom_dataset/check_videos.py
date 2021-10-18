import os


dir1 = "/media/daton/D6A88B27A88B0569/dataset/ava/train/videos"
dir2 = "/media/daton/D6A88B27A88B0569/dataset/ava/train/videos_15min"

files1 = os.listdir(dir1)
files2 = os.listdir(dir2)

inter = set(files1) & set(files2)
print(inter)
print(len(inter))

for f in inter:
    file_path = os.path.join(dir1, f)
    if os.path.isfile(file_path):
        pass
        #os.remove(file_path)