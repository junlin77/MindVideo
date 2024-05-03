import cv2
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
from lavis.models import load_model_and_preprocess

def read_videos_and_process(directory_path, num_videos=18, num_clips=240, frames_per_clip=6, frame_height=256, frame_width=256):
    """
    Output shape: (18, 240, 6, 256, 256, 3), 18 videos, 240 clips per video, 6 frames per clip (3 fps x 2 seconds)
    """
    # Initialize the numpy array to store processed video data
    video_data = np.zeros((num_videos, num_clips, frames_per_clip, frame_height, frame_width, 3), dtype=np.uint8)
    
    video_index = 0
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".mp4") or filename.endswith(".avi"):  # Assuming video files are in mp4 or avi format
            video_path = os.path.join(directory_path, filename)
            cap = cv2.VideoCapture(video_path)
            
            clip_index = 0
            frame_count = 0
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to desired resolution
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Append frame to list of frames
                frames.append(frame)
                frame_count += 1
                
                # If we have collected enough frames for a clip, process and save it
                if frame_count == frames_per_clip:
                    video_data[video_index, clip_index, :, :, :, :] = np.array(frames)
                    frames = []
                    clip_index += 1
                    frame_count = 0
                    
                    # If we have collected enough clips for a video, move to the next video
                    if clip_index == num_clips:
                        video_index += 1
                        break
            
            cap.release()
            
            # If we have collected enough videos, stop processing
            if video_index == num_videos:
                break
    
    # Save the processed video data to a single .npy file
    np.save('/Volumes/dfdf/Research/Scene Reconstruction/Data/cc2017/processed_videos.npy', video_data)

def caption_videos(directory_path):
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BLIP captioning model
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # Initialize list to store video captions
    video_captions = []

    # Iterate over video files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory_path, filename)

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Process each frame in the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get image from the frame 
                raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                # generate caption
                caption = model.generate({"image": image})

                # Store the caption for the current frame
                video_captions.append(caption)

            cap.release()

    return video_captions

if __name__ == '__main__':
    directory_path = '/Volumes/dfdf/Research/Scene Reconstruction/Data/cc2017/10_4231_R71Z42KK/video_fmri_dataset/stimuli_3fps_256'

    # read_videos_and_process(directory_path)
    captions = caption_videos(directory_path)
    for i, caption in enumerate(captions):
        print(f"Frame {i + 1} Caption: {caption}")
