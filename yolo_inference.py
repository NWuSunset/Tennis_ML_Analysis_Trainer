from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.track("../data/input_video.mp4", save=True, persist=True)