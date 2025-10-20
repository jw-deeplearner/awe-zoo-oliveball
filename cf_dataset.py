from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from bespoke_frame_extractor import * 
from bespoke_ffmpeg_silencer import *
av = safe_import_av()
from collections import Counter

class CustomVideoDataset(Dataset):
    def __init__(self, classes_path, labels_dict, clip_length_in_frames=16, transforms=None, video_backend = "pyav",disable_progress_bar=False,decode_to_get_frame_count=False):
        self.clip_length_in_frames = clip_length_in_frames
        self.transforms = transforms
        self.classes_path = classes_path
        self.samples = []
        self.video_backend = video_backend 

        if self.video_backend == 'pyav':
            self.count_frames = count_frames_av
            self.decode_video = decode_video_av
        elif self.video_backend == "video_reader-rs":
            print("[WARNING]: Using the video_reader-rs backend is not recommended - it can't decode many videos properly (decodes as black frames) and is also significantly slower than PyAV.")
            self.count_frames = count_frames_rs
            self.decode_video = decode_video_rs
        else:
            raise ValueError(f"Unsupported video backend: {self.video_backend}")

        for video_path, label in tqdm(labels_dict.items(), desc="Initiating CustomVideoDataset", unit="video", colour='cyan', disable=disable_progress_bar,dynamic_ncols=True):
            full_path = self.classes_path / video_path
            parts = Path(video_path).parts
            if len(parts) < 3:
                continue

            major_category, minor_category, filename = parts[0], parts[1], parts[2]

            # Open video with PyAV to get info
            try:
                if decode_to_get_frame_count == True and self.video_backend == 'video_reader-rs':
                    print("[WARNING]: Using decoding to get frame count with video_reader-rs (PyVideoReader) is not supported and WILL result in inaccurate counts.")
                total_frames = self.count_frames(full_path,decode_to_get_frame_count)
            except Exception as e:
                print(f"[WARNING]: Skipping video {video_path}: failed to open using backend {self.video_backend} ({e})")
                continue

            # Choose stride based on frame count
            if total_frames > 3.375*self.clip_length_in_frames:    # default with 16-len clips: 54
                stride = 3
            elif total_frames >= 2*self.clip_length_in_frames:     # default with 16-len clips: 32
                stride = 2
            elif total_frames >= self.clip_length_in_frames:       # default with 16-len clips: 16
                stride = 1
            elif total_frames >= 0.375*self.clip_length_in_frames: # default with 16-len clips: 6
                stride = (total_frames - 1) / (clip_length_in_frames - 1)
            else:
                print(f"Throwing out video {video_path}, too short! Only {total_frames} long.")
                continue

            effective_len = round(self.clip_length_in_frames * stride)
            max_start = total_frames - effective_len
            if max_start < 0:
                print(f"Skipping video {video_path}: sample starts with negative index: {max_start}")
                continue
            
            #Yields maximum 50% overlap between multiple clips being extracted (minimise overfitting, maximise clips produced from video)
            step = max(1, int(effective_len // 2))  # make sure step is at least 1

            for start_idx in range(0, max(1, int(max_start) + 1), step):

                clip_would_go_out_of_bounds = start_idx + effective_len > total_frames
                #Realistically should never be called but good to have as a sanity check
                if clip_would_go_out_of_bounds:
                    print("Warning: Clip would have gone out of bounds")
                    start_idx = total_frames - effective_len

                #Note if you change this please update cf_glance.py to write train/test to _dataset_samples.json properly (keys are manual)
                self.samples.append((
                    video_path,
                    start_idx,
                    stride,
                    label,
                    major_category,
                    minor_category,
                    filename,
                    total_frames 
                ))

                if clip_would_go_out_of_bounds:
                    break

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start_idx, stride, label, major_category, minor_category, filename, total_frames = self.samples[idx]

        indices = get_frame_indices(start_idx, stride, total_frames, self.clip_length_in_frames)
        try:
            frames = self.decode_video(self.classes_path / video_path, indices)  # [T, H, W, C]
        except FileNotFoundError:
            print(f"[WARNING] File not found @path <{video_path}>\nVideo replaced with black frames.")
            black_video = torch.zeros((self.clip_length_in_frames, 3, 576, 576))
            return black_video, -100, {"video_path": str(video_path), "error": "File not found"}
        
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        if self.transforms:
            frames_tensor = self.transforms(frames_tensor)

        metadata = {
            "video_path": video_path,
            "filename": filename,
            "major_category": major_category,
            "minor_category": minor_category,
            "start_frame": start_idx,
            "stride": stride,
            "num_frames": len(indices),
            "indices": indices.tolist()
        }

        return frames_tensor, label, metadata


def generate_class_weights(video_dataset, compute_device, normalize=False):
    """
    Compute class weights from a CustomVideoDataset.

    Args:
        video_dataset: CustomVideoDataset object (must have .samples with label at index 3).
        compute_device: torch.device ("cpu", "cuda", or "mps").
        normalize: if True, scale weights to have mean = 1.

    Returns:
        torch.FloatTensor of shape [num_classes]
    """
    # Extract all class indices from dataset samples
    all_class_labels = [sample[3] for sample in video_dataset.samples]

    # Count how many samples belong to each class
    class_sample_counts = Counter(all_class_labels)
    #print(class_sample_counts.items())

    # Total number of unique classes in this dataset split
    number_of_classes = len(set(all_class_labels))

    # Tensor to hold counts, indexed by class
    class_counts_tensor = torch.zeros(number_of_classes, dtype=torch.float)
    for class_index, sample_count in class_sample_counts.items():
        class_counts_tensor[class_index] = sample_count

    # Inverse frequency weighting (rare classes get higher weight)
    class_weights = 1.0 / class_counts_tensor

    # Optionally normalize weights to have mean = 1
    if normalize:
        class_weights = class_weights / class_weights.sum() * number_of_classes

    return class_weights.to(compute_device)