import os
import shutil

import numpy as np
from tqdm import tqdm


target_dir = "/media/daton/D6A88B27A88B0569/dataset/military_civil2/train/civil"
out_dir = "/media/daton/D6A88B27A88B0569/dataset/military_civil2/valid/civil"
target_files = os.listdir(target_dir)
rm_idx = np.random.randint(0, len(target_files), 10000)

for i, file in enumerate(tqdm(target_files)):
    file_path = os.path.join(target_dir, file)
    if i in rm_idx:
        out_path = os.path.join(out_dir, file)
        shutil.move(file_path, out_path)
        pass