import argparse
import os

import cv2
import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args():
    default_image_folder = os.environ.get(
        "DREAM_IMAGE_FOLDER",
        os.path.join(SCRIPT_DIR, "figsave_DREAM_inD_benchmark_03"),
    )
    default_video_name = os.environ.get(
        "DREAM_VIDEO_NAME",
        os.path.join(default_image_folder, "inD_dream_benchmark_03.mp4"),
    )

    parser = argparse.ArgumentParser(
        description="Create MP4 from numeric PNG frame sequence."
    )
    parser.add_argument(
        "--image-folder",
        default=default_image_folder,
        help="Folder with numeric PNG frames (0.png, 1.png, ...).",
    )
    parser.add_argument(
        "--video-name",
        default=default_video_name,
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video FPS.")
    parser.add_argument("--max-width", type=int, default=1920, help="Max output width.")
    parser.add_argument("--max-height", type=int, default=1080, help="Max output height.")
    parser.add_argument(
        "--letterbox",
        type=_str2bool,
        default=True,
        help="Keep aspect ratio by adding black bars if needed.",
    )
    return parser.parse_args()


def _even(v):
    return int(v) if int(v) % 2 == 0 else int(v) - 1


def _fit_size(src_w, src_h, max_w, max_h):
    scale = min(max_w / float(src_w), max_h / float(src_h), 1.0)
    out_w = max(2, _even(round(src_w * scale)))
    out_h = max(2, _even(round(src_h * scale)))
    return out_w, out_h


def _letterbox(frame, out_w, out_h):
    h, w = frame.shape[:2]
    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def main():
    args = _parse_args()
    image_folder = os.path.abspath(args.image_folder)
    video_name = os.path.abspath(args.video_name)
    os.makedirs(os.path.dirname(video_name), exist_ok=True)

    images = [
        img for img in os.listdir(image_folder)
        if img.endswith(".png") and os.path.splitext(img)[0].isdigit()
    ]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if not images:
        raise FileNotFoundError(f"No numeric PNG frames found in {image_folder}")

    first = cv2.imread(os.path.join(image_folder, images[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {images[0]}")

    src_h, src_w = first.shape[:2]
    out_w, out_h = _fit_size(src_w, src_h, args.max_width, args.max_height)

    print(f"Frames: {len(images)}  |  FPS: {args.fps}")
    print(f"Source frame: {src_w}x{src_h}")
    print(f"Output video: {out_w}x{out_h}  (letterbox={args.letterbox})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_name, fourcc, args.fps, (out_w, out_h))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is None:
            print(f"[WARN] Cannot read {image}, skipping.")
            continue

        if args.letterbox:
            frame_out = _letterbox(frame, out_w, out_h)
        else:
            frame_out = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame_out)

    writer.release()
    print(f"Video created: {video_name}")


if __name__ == "__main__":
    main()
