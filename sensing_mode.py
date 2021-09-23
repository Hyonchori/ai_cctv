import numpy as np
import cv2
from collections import deque


class RoI():
    def __init__(self,
                 img_size: tuple=None,
                 window_name: str="roi",
                 default_roi: np.ndarray=None):
        self.img_size = img_size
        self.window_name = window_name
        self.default_roi = default_roi

        self.use_default_roi = True if default_roi is not None else False
        self.ref_img = np.zeros(img_size, np.uint8)
        self.roi_pts = [] if default_roi is None else default_roi
        self.roi_check = False if not self.roi_pts else True
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.roi_check:
                self.roi_check = False
                self.roi_pts = []
                self.ref_img = np.zeros(self.img_size, np.uint8)
            elif not self.roi_check and len(self.roi_pts) == 0:
                pass
            else:
                self.roi_check = True
                self.roi_pts = np.array(self.roi_pts, np.int32)

    def imshow(self, input_img):
        roi_pts = self.roi_pts
        if not self.roi_check:
            for pt in roi_pts:
                cv2.circle(input_img, (pt[0], pt[1]), 2, (0, 225, 225), 2)
        else:
            cv2.fillPoly(self.ref_img, [roi_pts.reshape((-1, 1, 2))], (0, 225, 225))

        result = cv2.addWeighted(input_img, 1, self.ref_img, 0.5, 0)
        cv2.imshow(self.window_name, result)
        return result


class SenseTrespassing(RoI):
    def __init__(self,
                 colors,
                 img_size: tuple = None,
                 window_name: str = "sensing_trespassing",
                 default_roi: np.ndarray = None):
        super().__init__(img_size, window_name, default_roi)
        self.colors = colors
        self.sensed_person = []
        self.just_person = []

    def update(self, dets, person_cls=0):
        if not self.roi_check:
            pass
        else:
            for *xyxy, conf, cls in reversed(dets):
                c = int(cls)
                if not c == person_cls:
                    continue
                foot_pt = self.xyxy2foot(xyxy)
                dist = cv2.pointPolygonTest(self.roi_pts, foot_pt, True)
                if dist > 0:
                    self.sensed_person.append(xyxy)
                else:
                    self.just_person.append(xyxy)

    def xyxy2foot(self, xyxy, foot_height=0.05):
        x_min = xyxy[0].item()
        y_min = xyxy[1].item()
        x_max = xyxy[2].item()
        y_max = xyxy[3].item()
        height = y_max - y_min
        return [int((x_min + x_max) / 2), int(y_max - foot_height * height)]

    def imshow(self, input_img):
        roi_pts = self.roi_pts
        sensed = True if self.sensed_person else False
        if not self.roi_check:
            for pt in roi_pts:
                cv2.circle(input_img, (pt[0], pt[1]), 2, (0, 225, 225), 2)
        else:
            if sensed:
                cv2.fillPoly(self.ref_img, [roi_pts.reshape((-1, 1, 2))], (0, 0, 225))
            cv2.drawContours(self.ref_img, [roi_pts.reshape((-1, 1, 2))], 0, (0, 0, 255), 2)

        result = cv2.addWeighted(input_img, 1, self.ref_img, 0.5, 0)

        for jp in self.just_person:
            self.draw_bbox(result, jp, False)
        for sp in self.sensed_person:
            self.draw_bbox(result, sp, True)

        cv2.imshow(self.window_name, result)

        self.sensed_person = []
        self.just_person = []
        return result

    def draw_bbox(self, img, bbox, sensed, font_scale=1, fonst_thick=1):
        tmp_img = np.zeros(self.img_size, np.uint8)
        tl = (int(bbox[0]), int(bbox[1]))
        br = (int(bbox[2]), int(bbox[3]))
        if sensed:
            label = "Trepassing"
            color = self.colors(0, True)
        else:
            label = "person"
            color = self.colors(0 + 4, True)
        line_color = (max(color[0] - 30, 0), max(color[1] - 30, 0), max(color[2] - 30, 0))
        cv2.rectangle(tmp_img, tl, br, color, -1)
        cv2.rectangle(tmp_img, tl, br, line_color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, fonst_thick)[0]
        cv2.rectangle(
            tmp_img, (tl[0] - 1, tl[1] - 2),
            (int(bbox[0]) + t_size[0] + 2, int(bbox[1]) + t_size[1] + 3), line_color, -1
        )
        cv2.putText(tmp_img, label, (tl[0], tl[1] + 9),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, [255, 255, 255], fonst_thick, lineType=cv2.LINE_AA)
        tmp_result = cv2.addWeighted(img, 0.8, tmp_img, 0.4, 0)
        cv2.copyTo(tmp_result, tmp_img, img)




class SenseLoitering(RoI):
    def __init__(self,
                 colors, fps, max_buffer_size, max_count: int=10, time_thr: int=10,
                 img_size: tuple = None,
                 window_name: str = "sensing_loitering",
                 default_roi: np.ndarray = None):
        super().__init__(img_size, window_name, default_roi)
        self.colors = colors
        self.fps = fps
        self.max_buffer_size = max_buffer_size
        self.time_thr = time_thr
        self.loitering_count = {k: deque([]) for k in range(1, max_count+1)}
        self.sensed_id = []
        self.just_id = []

    def check_track(self, track):
        if not self.roi_check:
            pass
        else:
            id = track[-1]
            foot_pt = self.xyxy2foot(track)


    def xyxy2foot(self, xyxy, foot_height=0.05):
        x_min = xyxy[0].item()
        y_min = xyxy[1].item()
        x_max = xyxy[2].item()
        y_max = xyxy[3].item()
        height = y_max - y_min
        return [int((x_min + x_max) / 2), int(y_max - foot_height * height)]