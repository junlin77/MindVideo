import cv2
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
from lavis.models import load_model_and_preprocess

def process_videos(directory_path, num_videos=18, num_clips=240, frames_per_clip=6, frame_height=256, frame_width=256):
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

def caption_videos(directory_path, output_file, num_videos=18, num_clips=240, frames_per_clip=6):
    """
    Generate captions for frames in videos and save the captions to a file.

    Args:
        directory_path (str): Path to the directory containing video files.
        output_file (str): Path to the output file to save the captions.
        num_videos (int): Number of videos to process.
        num_clips (int): Number of clips per video.
        frames_per_clip (int): Number of frames per clip.

    Returns:
        np.ndarray: Array of captions with shape (num_videos, num_clips, frames_per_clip).
    """
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BLIP captioning model
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # Initialize array to store captions
    captions_array = np.empty((num_videos, num_clips, frames_per_clip), dtype=object)

    video_index = 0
    clip_index = 0
    frame_count = 0

    # Iterate over video files in the directory
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory_path, filename)

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            frames = []

            # Process each frame in the video
            while cap.isOpened():
                ret, frame = cap.read()

                frame_count += 1
                if not ret:
                    break

                # Get image from the frame
                raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Preprocess the image
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                # Generate caption
                caption = model.generate({"image": image})

                print(f'Video: {video_index} | Clip: {clip_index} | Frame: {frame_count} | Caption: {caption}')

                # Store the caption for the current frame
                captions_array[video_index, clip_index, frame_count - 1] = caption

                # If we have collected enough frames for a clip, move to the next clip
                if frame_count == frames_per_clip:
                    clip_index += 1
                    frame_count = 0

                    # If we have collected enough clips for a video, move to the next video
                    if clip_index == num_clips:
                        video_index += 1
                        clip_index = 0

                        # If we have processed enough videos, stop processing
                        if video_index == num_videos:
                            break

            cap.release()

            # If we have processed enough videos, stop processing
            if video_index == num_videos:
                break

    # Save the generated captions to the output file
    np.save(output_file, captions_array)

    return captions_array

if __name__ == '__main__':
    directory_path = '/content/drive/MyDrive/Scene Reconstruction/stimuli_3fps'
    output_file = '/content/drive/MyDrive/Scene Reconstruction/text_train_256_3hz.npy'

    # Generate captions and save to file
    captions = caption_videos(directory_path, output_file)

    # Output captions
    for video_idx in range(captions.shape[0]):
        for clip_idx in range(captions.shape[1]):
            for frame_idx in range(captions.shape[2]):
                caption = captions[video_idx, clip_idx, frame_idx]
                print(f"Video {video_idx + 1}, Clip {clip_idx + 1}, Frame {frame_idx + 1} Caption: {caption}")
