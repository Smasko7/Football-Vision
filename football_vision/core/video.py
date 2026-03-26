"""Video I/O helpers — thin wrappers around the supervision library."""

import supervision as sv


def get_video_info(source_path: str) -> sv.VideoInfo:
    return sv.VideoInfo.from_video_path(source_path)


def make_video_sink(target_path: str, video_info: sv.VideoInfo) -> sv.VideoSink:
    return sv.VideoSink(target_path, video_info)


def get_frame_generator(source_path: str, stride: int = 1, start: int = 0):
    """Return a supervision frame generator with optional stride and start offset."""
    return sv.get_video_frames_generator(source_path, stride=stride, start=start)
