from utils import process_video
from trackers import PlayerTracker 

def main():
    # read input video
    input_video_path = "data/input_video.mp4"
    input_vido_long_path = "data/yt_amateur_tennis.mp4"
    output_video_path = "output_videos/output_video.avi"

    player_tracker = PlayerTracker(model_path = "yolo26n.pt")

    def process_frame(frame): #process one frame at a time to avoid memory issues. (could change to chunked frames)
        player_detections = player_tracker.detect_frame(frame)
        return player_tracker.draw_bboxes([frame], [player_detections])[0]

    process_video(input_video_path, output_video_path, process_frame)

if __name__ == "__main__":
    main()
