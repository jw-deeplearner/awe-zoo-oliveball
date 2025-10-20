def check_file_is_video(file):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    if file.suffix.lower() in video_extensions and file.name.startswith("._")==False:
        return True
    return False