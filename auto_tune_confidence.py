"""
Auto-tune Confidence Threshold
Find optimal confidence threshold for new videos
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path


def sample_frames(video_path, num_samples=10):
    """
    Sample frames evenly from video
    
    Args:
        video_path: Path to video
        num_samples: Number of frames to sample
        
    Returns:
        List of sampled frames
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sample indices
    indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    
    cap.release()
    return frames


def evaluate_confidence(model, frames, conf_threshold, device='cpu'):
    """
    Evaluate detection performance at given confidence threshold
    
    Args:
        model: YOLO model
        frames: List of (frame_id, frame)
        conf_threshold: Confidence threshold to test
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    total_detections = 0
    frames_with_detections = 0
    avg_conf = []
    
    for frame_id, frame in frames:
        results = model(frame, conf=conf_threshold, device=device, verbose=False)
        
        if results[0].obb is not None and len(results[0].obb) > 0:
            detections = len(results[0].obb)
            total_detections += detections
            frames_with_detections += 1
            
            # Get confidences
            for obb in results[0].obb:
                conf = float(obb.conf[0])
                avg_conf.append(conf)
    
    return {
        'total_detections': total_detections,
        'frames_with_detections': frames_with_detections,
        'avg_detections_per_frame': total_detections / len(frames) if len(frames) > 0 else 0,
        'avg_confidence': np.mean(avg_conf) if len(avg_conf) > 0 else 0,
        'detection_rate': frames_with_detections / len(frames) if len(frames) > 0 else 0
    }


def find_optimal_confidence(video_path, model_path='yolov8n-obb.pt', 
                           conf_range=(0.3, 0.7), num_steps=9,
                           num_sample_frames=10, device='auto',
                           target_detections_per_frame=2.0):
    """
    Find optimal confidence threshold for video
    
    Args:
        video_path: Path to video
        model_path: Path to YOLO model
        conf_range: Range of confidence to test (min, max)
        num_steps: Number of steps to test
        num_sample_frames: Number of frames to sample
        device: 'auto', 'cuda', or 'cpu'
        target_detections_per_frame: Expected number of drones per frame
        
    Returns:
        Optimal confidence threshold
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("🎯 AUTO-TUNING CONFIDENCE THRESHOLD")
    print("="*70)
    print(f"📹 Video: {Path(video_path).name}")
    print(f"🔧 Device: {device}")
    print(f"📊 Testing range: {conf_range[0]} - {conf_range[1]}")
    print(f"🎬 Sample frames: {num_sample_frames}")
    print(f"🎯 Target detections/frame: {target_detections_per_frame}")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Sample frames
    print(f"🎬 Sampling {num_sample_frames} frames...")
    frames = sample_frames(video_path, num_sample_frames)
    print(f"   ✅ Sampled {len(frames)} frames")
    
    # Test confidence thresholds
    print(f"\n🧪 Testing {num_steps} confidence thresholds...")
    print("-"*70)
    print(f"{'Conf':<8} {'Detections':<12} {'Det/Frame':<12} {'Avg Conf':<12} {'Det Rate':<12}")
    print("-"*70)
    
    thresholds = np.linspace(conf_range[0], conf_range[1], num_steps)
    results = []
    
    for conf in thresholds:
        metrics = evaluate_confidence(model, frames, conf, device)
        results.append({
            'conf': conf,
            **metrics
        })
        
        print(f"{conf:<8.2f} {metrics['total_detections']:<12} "
              f"{metrics['avg_detections_per_frame']:<12.2f} "
              f"{metrics['avg_confidence']:<12.2f} "
              f"{metrics['detection_rate']:<12.2f}")
    
    # Find optimal threshold
    # Strategy: Find threshold closest to target detections/frame
    # while maintaining high detection rate (>50% of frames)
    
    print("-"*70)
    print("\n🔍 Finding optimal threshold...")
    
    valid_results = [r for r in results if r['detection_rate'] >= 0.5]
    
    if not valid_results:
        print("⚠️  Warning: Low detection rate across all thresholds")
        print("   Using threshold with highest detection rate")
        optimal = max(results, key=lambda x: x['detection_rate'])
    else:
        # Find closest to target
        optimal = min(valid_results, 
                     key=lambda x: abs(x['avg_detections_per_frame'] - target_detections_per_frame))
    
    print("\n" + "="*70)
    print("✅ OPTIMAL CONFIDENCE FOUND")
    print("="*70)
    print(f"🎯 Optimal confidence: {optimal['conf']:.2f}")
    print(f"📊 Total detections: {optimal['total_detections']}")
    print(f"📈 Detections per frame: {optimal['avg_detections_per_frame']:.2f}")
    print(f"💯 Average confidence: {optimal['avg_confidence']:.2f}")
    print(f"✅ Detection rate: {optimal['detection_rate']:.1%}")
    print("="*70)
    
    return round(optimal['conf'], 2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-tune confidence threshold')
    parser.add_argument('--video', type=str, required=True,
                       help='Video file')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='YOLO model path')
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help='Minimum confidence to test')
    parser.add_argument('--max-conf', type=float, default=0.7,
                       help='Maximum confidence to test')
    parser.add_argument('--steps', type=int, default=9,
                       help='Number of steps')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of frames to sample')
    parser.add_argument('--target', type=float, default=2.0,
                       help='Target detections per frame')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu')
    
    args = parser.parse_args()
    
    optimal_conf = find_optimal_confidence(
        video_path=args.video,
        model_path=args.model,
        conf_range=(args.min_conf, args.max_conf),
        num_steps=args.steps,
        num_sample_frames=args.samples,
        device=args.device,
        target_detections_per_frame=args.target
    )
    
    print(f"\n💡 Recommended usage:")
    print(f"   python problem1_competition.py --video {args.video} --conf {optimal_conf}")
    print(f"   python problem3_integration.py --video {args.video} --conf {optimal_conf}")
