# future todo: separate pose estimation from bounding box for opposing player, we don't need pose for the far player generally, just their position.
import cv2
import pickle
from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame): 
        results = self.model.track(frame, persist=True)[0] 
        class_names = results.names 
        player_dict = {} #save ids and bounding box coords

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0] #bounding box coords 

            class_ids = box.cls.tolist()[0] # class ids for the detected bounding box
            det_class_names = class_names[class_ids] #get corresponding name for the id

            if det_class_names == "person":
                player_dict[track_id] = result

        return player_dict

    def detect_frames(self, frames):
        player_detections = [] #list of player dictionaries (which store ids and bounding box coordinates)

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        output_frames = []

        for frame, player_dict in zip(video_frames, player_detections): #loop through frames (and detections per frame)
            for track_id, bbox in player_dict.items(): #draw bounding boxes around players
                x1, y1, x2, y2 = bbox #box.xyxy
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            output_frames.append(frame)

        return output_frames