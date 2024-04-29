import os
import cv2
import numpy as np

def process_videos_to_npy(video_dir, output_path, num_videos=18, num_frames=240, num_subframes=6, frame_size=(256, 256)):
    # Initialize an empty array to hold the processed video data
    video_data = np.zeros((num_videos, num_frames, num_subframes, frame_size[0], frame_size[1], 3), dtype=np.uint8)

    for vid_idx in range(1,18):
        print(f"Processing seg{vid_idx}.mp4")
        video_path = os.path.join(video_dir, f'seg{vid_idx}.mp4')

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine indices to sample frames evenly across the video
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

        # Iterate over the sampled frame indices
        for frame_idx, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            # Resize the frame to the desired size
            resized_frame = cv2.resize(frame, frame_size)

            # Divide the resized frame into subframes
            for subframe_idx in range(num_subframes):
                subframe_width = frame_size[0] // num_subframes
                subframe = resized_frame[:, subframe_idx * subframe_width:(subframe_idx + 1) * subframe_width, :]

                # Store the subframe in the video data array
                video_data[vid_idx, frame_idx, subframe_idx] = subframe

        # Release the video capture object
        cap.release()

    # Save the processed video data to a .npy file
    np.save(output_path, video_data)

    print(f"Video data saved to {output_path} with shape {video_data.shape}")

if __name__ == '__main__':
    # Specify input directory containing the video files
    input_video_dir = '/Volumes/dfdf/Research/Scene Reconstruction/Data/cc2017/10_4231_R71Z42KK/video_fmri_dataset/stimuli_3fps_256'

    # Specify output path for the .npy file
    output_npy_path = '/Volumes/dfdf/Research/Scene Reconstruction/Data/cc2017/video_train_256_3hz.npy'

    # Process videos and save the output to .npy file
    process_videos_to_npy(input_video_dir, output_npy_path)