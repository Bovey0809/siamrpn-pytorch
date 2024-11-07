
from siamrpn import TrackerSiamRPN
import cv2

net_path = 'pretrained/siamrpn/model.pth'
tracker = TrackerSiamRPN(net_path=net_path)

# extract video frames
video_path = 'demo.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# inference
boxes, times = tracker.track(frames, box=[100, 100, 100, 100], visualize=True)
print(boxes)
print(times)
