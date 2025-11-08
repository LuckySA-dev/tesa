"""
Make a simple sample video from a single image by applying slight motions.
This helps test Problem 3 tracking when you don't have a real video.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def make_video(image_path: str, output_path: str, frames: int = 120, fps: int = 30,
               move_px: int = 20, rotate_deg: float = 5.0, scale_amp: float = 0.03):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # Prepare writer
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Animate affine transforms to simulate slight motion
    for t in range(frames):
        # Oscillating translation
        dx = int(move_px * np.sin(2 * np.pi * t / frames))
        dy = int(move_px * np.cos(2 * np.pi * t / frames))

        # Oscillating rotation and scale
        angle = rotate_deg * np.sin(2 * np.pi * t / frames)
        scale = 1.0 + scale_amp * np.sin(4 * np.pi * t / frames)

        # Build transformation matrix
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        M[:, 2] += [dx, dy]

        frame = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        writer.write(frame)

    writer.release()
    print(f"Saved sample video to: {output_path}  ({frames} frames @ {fps} FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample video from a single image")
    parser.add_argument('--input', type=str, default='images/drones.jpg', help='Input image path')
    parser.add_argument('--output', type=str, default='videos/video_01.mp4', help='Output video path')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--move', type=int, default=20, help='Max translation in pixels')
    parser.add_argument('--rotate', type=float, default=5.0, help='Max rotation in degrees')
    parser.add_argument('--scale', type=float, default=0.03, help='Scale amplitude')
    args = parser.parse_args()

    make_video(args.input, args.output, args.frames, args.fps, args.move, args.rotate, args.scale)
