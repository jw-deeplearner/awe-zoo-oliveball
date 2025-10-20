import imageio
import torch
from torch.nn import functional as F
from pathlib import Path
from torch.utils.data import default_collate
from enum import Enum
from bespoke_tools import * 
from bespoke_ffmpeg_silencer  import silent_import
transforms = silent_import("torchvision.transforms.v2")

def serialise_value(value):
    if isinstance(value, Enum):
        return str(value)
    elif isinstance(value, (list, tuple)):
        return [serialise_value(item) for item in value]
    elif isinstance(value, dict):
        return {key: serialise_value(val) for key, val in value.items()}
    else:
        return value
    
def custom_collate(batch):
    batch_frames = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    batch_metadata = [item[2] for item in batch]
    return default_collate(batch_frames), default_collate(batch_labels), batch_metadata

class PermuteCTHWtoTCHW(torch.nn.Module):
    def forward(self, x):
        # [C,T,H,W] → [T,C,H,W]
        return x.permute(1, 0, 2, 3)

def get_normalize_mean_std(dataset):
    # Safely get the list of transforms (supporting Compose or direct transforms attr)

    transforms_list = getattr(dataset.transforms, 'transforms', None)
    if not transforms_list:
        if getattr(dataset.transforms,'mean',None) is not None and getattr(dataset.transforms, 'std',None) is not None:
            return dataset.transforms.mean, dataset.transforms.std
        return None, None
    
    for transform in transforms_list:
        if hasattr(transform,"mean") and hasattr(transform, "std"):
            return transform.mean, transform.std
    return None, None

def extract_transforms_info(transforms_pipeline):
    transform_steps = getattr(transforms_pipeline, 'transforms', None)
    if not transform_steps:
        transform_steps = [transforms_pipeline]

    transforms_info = []
    for transform_step in transform_steps:
        transform_name = transform_step.__class__.__name__
        transform_parameters = {k: serialise_value(v) for k, v in getattr(transform_step, '__dict__', {}).items() if not k.startswith('_')}
        transforms_info.append({
            'name': transform_name,
            'parameters': transform_parameters,
        })
    return transforms_info

def extract_batch_evaluation_metadata(outputs, batch_size, batch_labels, batch_metadata, labels_index):

    probabilities = F.softmax(outputs, dim=1)
    k = min(5, outputs.size(1))
    top5_probabilities, top5_indices = probabilities.topk(k, dim=1)

    batch_evaluation_metadata = []
    for i in range(batch_size):
        true_index = int(batch_labels[i].item())
        predicted_index = int(torch.argmax(outputs[i]).item())

        true_label_name = get_first_key_from_value(labels_index, true_index)
        predicted_label_name = get_first_key_from_value(labels_index, predicted_index)

        prediction_confidence = float(probabilities[i, predicted_index].item())

        top5_labels = []
        top5_values = []
        for j in range(k):
            label_name = get_first_key_from_value(labels_index, int(top5_indices[i, j].item()))
            probability_value = float(top5_probabilities[i, j].item())
            top5_labels.append(label_name)
            top5_values.append(probability_value)

        clip_metadata = batch_metadata[i]
        clip_evaluation_metadata = {
            **clip_metadata,
            "true_label_name": true_label_name,
            "predicted_label_name": predicted_label_name,
            "prediction_confidence": prediction_confidence,
            "top5_labels": top5_labels,
            "top5_probabilities": top5_values,
        }

        batch_evaluation_metadata.append(clip_evaluation_metadata)

    return batch_evaluation_metadata
    
def save_batch_gif(batch_frames, batch_metadata, batch_labels, labels_index, mean, std, output_dir: str | Path, assumed_fps=25,should_print_output=False):
    """
    Saves each video clip in a batch as a GIF, using stride-adjusted fps and filenames
    that include the original video filename.

    Args:
        batch_frames (Tensor): Batch of video clips, shape [B, T, C, H, W]
        batch_metadata (list of dict): List of metadata dicts for each sample
        mean (list): Normalization mean
        std (list): Normalization std
        output_dir (str or Path): Directory to save GIFs
        assumed_fps (int): Base video fps before striding (default 25)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    gifs_metadata = {}

    for i, (clip, metadata, label) in enumerate(zip(batch_frames, batch_metadata, batch_labels), start=1):

        stride = metadata.get("stride")
        fps = max(1, assumed_fps // stride)

        original_filename = Path(metadata["filename"]).stem
        label_name = get_first_key_from_value(labels_index,label.item())
        filename = output_dir / f"quick_view_batch{i}-{original_filename}-{label_name}.gif"  # .item() for scalar
        save_gif(clip, filename, mean, std, fps=fps)
        if should_print_output:
            print(f"Saved {filename} @{fps}FPS (stride={stride})")

        gifs_metadata[f"gif{i}"] = {
            "output_file": str(filename),
            "filename": str(metadata.get("filename")),
            "label": label_name,
            "stride": stride,
            "assumed_source_fps": assumed_fps,
            "reconstructed_fps": fps,
            "total_frames": metadata.get("num_frames", "unknown"),
            "start_frame":metadata.get("start_frame"),
            "indices":metadata.get("indices")
        }

        metadata_path = output_dir / "batch_metadata.json"
        write_dict_to_json(gifs_metadata, metadata_path,print_message=should_print_output)


def save_gif(clip_tensor, filename: Path, mean, std, fps=15):
    """
    Save a video clip tensor as a GIF file.

    Args:
        clip_tensor: Tensor of shape [T, C, H, W], normalized.
        filename: Path to save the GIF.
        mean: list or tensor of mean values per channel (3,)
        std: list or tensor of std values per channel (3,)
        fps: Frames per second for the GIF.
    """
    mean = torch.tensor(mean, device=clip_tensor.device).reshape(1, 3, 1, 1)
    std = torch.tensor(std, device=clip_tensor.device).reshape(1, 3, 1, 1)

    # Denormalize and clamp to [0,1]
    clip_denorm = clip_tensor * std + mean
    clip_denorm = clip_denorm.clamp(0, 1)

    # Convert to numpy [T, H, W, C] uint8
    clip_np = (clip_denorm.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
    black_frame_indices = [i for i, frame in enumerate(clip_np) if frame.max() == 0]
    if black_frame_indices:
        print(f"Warning: Fully black frames detected at indices: {black_frame_indices} in {filename.name}")

    # Save as GIF
    imageio.mimsave(filename, clip_np, fps=fps)