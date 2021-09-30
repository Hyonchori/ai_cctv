import cv2
import time


sources = [
          "/home/daton/Downloads/loitering demo mosiac.mp4",
          #"/home/daton/Downloads/trespassing demo.mp4",
          #"/home/daton/Downloads/keypoint demo.mp4"
          ]

caps = [cv2.VideoCapture(s) for s in sources]

for source, cap in zip(sources, caps):
    print("\n---")
    #cap = cv2.VideoCapture(source)

    save_path = f"expanded {source.split('/')[-1]}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    start = time.time()
    target_time = 250

    wrt = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while cv2.waitKey(33) < 0:
        ref, frame = cap.read()
        wrt.write(frame)
        cv2.imshow("img", frame)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            tmp = time.time()
            print(tmp - start)
            if tmp - start < target_time:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            else:
                break

    cap.release()
    cv2.destroyAllWindows()