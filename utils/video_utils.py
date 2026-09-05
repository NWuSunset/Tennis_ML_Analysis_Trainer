import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [] #list of frames

    try: 
        while cap.isOpened():
             ret, frame = cap.read()

             if not ret:
                 break

             frames.append(frame)
    finally:
        cap.release()
        return frames

def save_video(output_frames, output_path):
    if not output_frames:
        raise ValueError("Cannot save a video with no frames")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #specify codec, compression format, pixel/color format

    width = output_frames[0].shape[1]
    height = output_frames[0].shape[0]

    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame in output_frames:
        out.write(frame)
    out.release()


def process_video(input_path, output_path, process_frame):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame)) #write processed frame
    finally:
        cap.release()
        out.release()