import os
from pprint import pprint
from tqdm import tqdm

label_dir =  "/media/daton/D6A88B27A88B0569/dataset/crowdhuman/labels_front_unsure_abnormal/train"
out_dir = "/media/daton/D6A88B27A88B0569/dataset/crowdhuman/labels/test"

labels = sorted(os.listdir(label_dir))
abnormal_num = 0
normal_num = 0
for l in tqdm(labels):
    #print("\n---")
    label_path = os.path.join(label_dir, l)
    with open(label_path) as f:
        data = f.readlines()
        data = [x.replace("\n", "") for x in data]
        box_dict = {}
        for i, d in enumerate(data):
            tmp = d[2:]
            if tmp not in box_dict:
                box_dict[tmp] = [d[0]]
            else:
                box_dict[tmp].append(d[0])

        txt = ""
        for k, v in box_dict.items():
            if len(v) > 1:
                for head_label in v:
                    if head_label != "0":
                        ttt = head_label
                line = ttt + " " + k
            else:
                line = v[0] + " " + k
            txt += line + "\n"
        txt = txt[:-1]

        out_path = os.path.join(out_dir, l)
        with open(out_path, "w") as of:
            of.write(txt)

        if len(data) != len(box_dict):
            print(l)
            break

