import numpy as np
from pathlib import Path
from bespoke_ffmpeg_silencer  import safe_import_av, filter_ffmpeg_warnings_from_stderr, silent_import
import subprocess
transforms = silent_import("torchvision.transforms.v2")
#PyVideoReader = silent_import("video_reader").PyVideoReader
PyVideoReader = None
av = safe_import_av()

def get_frame_indices(start_idx, stride, total_frames, clip_length):
    if stride <= 0:
        raise ValueError(f"Listen budweiser, your {stride} makes absolutely no sense. Stride needs to be a positive value.")
    
    # Dynamically calculate stride for very short videos
    if start_idx + stride * (clip_length - 1) > total_frames - 1:
        # This is fallback code, ideally is never called if used within proper structure.
        # I have this in here in case this function is called elsewhere.
        # Doesn't return stride so updated stride becomes invisible - can always return as tuple but for now unnecessary.
        stride = (total_frames - 1 - start_idx) / (clip_length - 1) if total_frames > 1 else 1
        print("WARNING: Reassigned stride")

    # Generate float indices, evenly spaced
    indices = start_idx + stride * np.arange(clip_length)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)

    return indices

def count_frames_rs(video_path: str | Path, decode=False):
    """
    Count frames in a video using PyVideoReader.

    Args:
        video_path (str or Path): Path to the video file.
        decode (bool): (Unused; kept for API symmetry with count_frames_av)

    Returns:
        int: Total number of frames.
    """
    vr = PyVideoReader(str(video_path))
    info = vr.get_info()
    total_frames = int(info["frame_count"])
    if decode: 
        batch_size = 32
        total_frames = 0
        # Since we don't know the real count, just attempt to decode until failure.
        start = 0
        while True:
            indices = list(range(start, start + batch_size))
            frames = filter_ffmpeg_warnings_from_stderr(vr.get_batch,indices)
            if len(frames) == 0:
                break
            # If all frames in batch are black (max==0 for every frame), stop
            is_all_black = np.all([np.max(frame) == 0 for frame in frames])
            if is_all_black:
                break
            # If some frames are black and others aren't, count only the non-black frames
            valid = [np.max(frame) > 0 for frame in frames]
            total_frames += sum(valid)
            # If batch had < batch_size non-black frames, we likely reached end
            if sum(valid) < batch_size:
                break
            start += batch_size

    return total_frames

def count_frames_av(video_path: str | Path, decode=False):
    # Open video with PyAV to get info

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames is None or total_frames <= 0 or decode:
        # Fall back to manual count
        total_frames = sum(1 for _ in container.decode(stream))
        container.seek(0)
    container.close()

    return total_frames 

def count_packets_av(video_path: str | Path):
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    count = 0
    for packet in container.demux(video_stream):
        # Only count packets that belong to the video stream
        if packet.stream.index == video_stream.index and packet.is_keyframe is not None:
            count += 1
    container.close()
    return count

def count_packets_ffprobe(video_path: str | Path):
    video_path = str(video_path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video_path)
    ]
    output = subprocess.check_output(cmd, text=True)
    try:
        return int(output.strip())
    except Exception:
        return None

def count_frames_ffprobe(video_path: str | Path,decode=False):
    video_path = str(video_path)
    cmd = ["ffprobe","-v", "error","-select_streams", "v:0", "-show_entries", 
           "stream=nb_frames","-of", "default=nokey=1:noprint_wrappers=1", str(video_path)]
    if decode:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path)
        ]
    output = subprocess.check_output(cmd, text=True)
    try:
        return int(output.strip())
    except ValueError:
        return None

def decode_video_rs(video_path, indices):
    vr = PyVideoReader(str(video_path))
    frames = filter_ffmpeg_warnings_from_stderr(vr.get_batch,indices)

    return frames 

def decode_video_av(video_path, indices):
    """
    Decodes frames at the requested indices from the video using PyAV.
    Raises RuntimeError if any frame can't be decoded.
    Returns frames as a numpy array [T, H, W, C] in uint8.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    indices_set = set(indices)
    frame_map = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in indices_set:
            img = frame.to_ndarray(format="rgb24")
            frame_map[i] = img
            if len(frame_map) == len(indices):
                break
    container.close()
    # Raise error if any frame is missing
    output = []
    for idx in indices:
        if idx in frame_map:
            output.append(frame_map[idx])
        else:
            raise RuntimeError(
                f"decode_frames_pyav: Failed to decode frame at index {idx} "
                f"in {video_path}. Only decoded indices: {sorted(frame_map.keys())}."
            )
    return np.stack(output, axis=0)