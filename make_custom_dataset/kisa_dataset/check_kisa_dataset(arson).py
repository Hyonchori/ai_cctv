import os
import datetime

import xmltodict
import cv2


def time2hms(time):
    h, m, s = [int(x) for x in time.split(":")]
    return h, m, s


def check_one_video(root, vid_name, out_root, show=False, save=False, save_interval=200):
    label_name = vid_name.replace(".mp4", ".xml")
    label_path = os.path.join(root, label_name)
    with open(label_path) as f:
        label = xmltodict.parse(f.read())["KisaLibraryIndex"]["Library"]["Clip"]
    alarm = label["Alarms"]["Alarm"]
    vid_time = time2hms(label["Header"]["Duration"])
    start_time = time2hms(alarm["StartTime"])
    duration = time2hms(alarm["AlarmDuration"])
    vid_time_delta = datetime.timedelta(hours=vid_time[0], minutes=vid_time[1], seconds=vid_time[2])
    start_time_delta = datetime.timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
    duration_delta = datetime.timedelta(hours=duration[0], minutes=duration[1], seconds=duration[2])
    start_frame_rate = start_time_delta / vid_time_delta
    end_frame_rate = (start_time_delta + duration_delta) / vid_time_delta

    vid_path = os.path.join(root, vid_name)
    cap = cv2.VideoCapture(vid_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(frames * start_frame_rate)
    end_frame = int(frames * end_frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            cap.release()
            break
        if show:
            cv2.imshow("img", img)
            cv2.waitKey(1)
        if save and idx % save_interval == 0:
            save_name = vid_name.replace(".mp4", "") + f"_{idx:04}.jpg"
            save_path = os.path.join(out_root, save_name)
            cv2.imwrite(save_path, img)
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame == end_frame:
            cap.release()
            break
        idx += 1


def extract_image_from_videos(root, out_root):
    vid_names = [x for x in os.listdir(root) if x.endswith(".mp4")]
    for vid_name in vid_names:
        print(f"\n--- Extracting images from video '{vid_name}'")
        check_one_video(root, vid_name, out_root, save=True)


if __name__ == "__main__":
    root = "/media/daton/SAMSUNG1/210806_지능형DB/3. 연구개발분야/1. 해외환경(1500개)/6. 방화(75개)"
    out_root = "/media/daton/D6A88B27A88B0569/dataset/fire detection/kisa"
    vid_name = "C050105_001.mp4"

    #check_one_video(root, vid_name, out_root, save=True)

    extract_image_from_videos(root, out_root)
