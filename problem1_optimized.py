"""
Optimized Problem 1: Detection + Tracking with Performance Enhancements
Based on problem1_competition.py with speed optimizations
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import time
from typing import Optional
import argparse

from byte_track_wrapper import ByteTrackWrapper
from optimize_performance import PerformanceOptimizer


class OptimizedDroneDetector:
    """Optimized drone detection and tracking system"""
    
    def __init__(
        self,
        model_path: str = 'yolov8n-obb.pt',
        conf_threshold: float = 0.5,
        device: str = 'auto',
        optimize: bool = True,
        skip_frames: int = 1
    ):
        """
        Initialize optimized detector
        
        Args:
            model_path: Path to YOLO-OBB model
            conf_threshold: Confidence threshold
            device: 'auto', 'cuda', or 'cpu'
            optimize: Apply performance optimizations
            skip_frames: Process every Nth frame (1 = all frames)
        """
        print("="*70)
        print("🚀 OPTIMIZED DRONE DETECTION SYSTEM")
        print("="*70)
        
        # Load model
        print(f"📦 Loading YOLO-OBB: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.skip_frames = skip_frames
        
        # Apply optimizations
        if optimize:
            self.optimizer = PerformanceOptimizer()
            self.model = self.optimizer.optimize_model(self.model, half_precision=False)
            self.optimizer.enable_torch_optimizations()
            self.optimizer.apply_nms_optimization(self.model, iou_threshold=0.45)
        
        self.tracker = None
        self.detections = []
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_detections': 0,
            'inference_time': 0,
            'tracking_time': 0,
        }
    
    def process_video(
        self,
        video_path: str,
        output_csv: str,
        save_video: bool = False,
        output_video: Optional[str] = None
    ):
        """
        Process video with detection and tracking
        
        Args:
            video_path: Path to input video
            output_csv: Path to output CSV
            save_video: Save annotated video
            output_video: Path to output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video: {Path(video_path).name}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
        print(f"🎬 Total frames: {total_frames}")
        print(f"🎯 Confidence: {self.conf_threshold}")
        print(f"⏭️  Skip factor: {self.skip_frames} ({'all frames' if self.skip_frames == 1 else f'every {self.skip_frames}th frame'})")
        
        # Initialize tracker with video's FPS
        self.tracker = ByteTrackWrapper(frame_rate=fps)
        print(f"📍 Tracker: ByteTrack (FPS: {fps})")
        
        # Video writer
        writer = None
        if save_video:
            if output_video is None:
                output_video = f"output/{Path(video_path).stem}_optimized.mp4"
            Path(output_video).parent.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        print("\n" + "="*60)
        
        # Process frames
        frame_id = 0
        start_time = time.time()
        last_tracked_objects = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should skip this frame
            if frame_id % self.skip_frames != 0:
                # Use last tracked objects for skipped frames
                if last_tracked_objects:
                    for obj in last_tracked_objects:
                        self.detections.append({
                            'frame_id': frame_id,
                            'object_id': obj['object_id'],
                            'center_x': obj['center_x'],
                            'center_y': obj['center_y'],
                            'w': obj['w'],
                            'h': obj['h'],
                            'theta': obj['theta']
                        })
                
                self.stats['frames_skipped'] += 1
                frame_id += 1
                
                # Write frame to video
                if writer:
                    # Draw previous detections
                    frame_vis = frame.copy()
                    for obj in last_tracked_objects:
                        cx = int(obj['center_x'] * width)
                        cy = int(obj['center_y'] * height)
                        cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.putText(frame_vis, f"ID:{obj['object_id']}", 
                                  (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
                    writer.write(frame_vis)
                
                continue
            
            # Detection
            inf_start = time.time()
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                verbose=False
            )[0]
            self.stats['inference_time'] += time.time() - inf_start
            
            # Extract OBB detections
            frame_detections = []
            detection_data = []  # Store full detection info
            if results.obb is not None and len(results.obb) > 0:
                obb = results.obb
                boxes = obb.xywhr.cpu().numpy()  # [cx, cy, w, h, rotation]
                confs = obb.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    cx, cy, w, h, angle = box
                    
                    # For tracker: just centroids
                    frame_detections.append((cx, cy))
                    
                    # Store full detection info
                    detection_data.append({
                        'center': (cx, cy),
                        'size': (w, h),
                        'angle': angle,
                        'conf': float(conf)
                    })
            
            # Tracking
            track_start = time.time()
            tracked = self.tracker.update(frame_detections, frame)
            self.stats['tracking_time'] += time.time() - track_start
            
            # Store current frame's tracked objects
            last_tracked_objects = []
            
            # Match tracked IDs to detection data
            tracked_dict = {tuple(v): k for k, v in tracked.items()}
            
            # Save detections with tracking IDs
            for det_data in detection_data:
                cx, cy = det_data['center']
                centroid_key = (cx, cy)
                
                if centroid_key in tracked_dict:
                    track_id = tracked_dict[centroid_key]
                    w, h = det_data['size']
                    angle = det_data['angle']
                    
                    # Normalize coordinates
                    cx_norm = cx / width
                    cy_norm = cy / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    detection = {
                        'frame_id': frame_id,
                        'object_id': track_id,
                        'center_x': cx_norm,
                        'center_y': cy_norm,
                        'w': w_norm,
                        'h': h_norm,
                        'theta': angle
                    }
                    
                    self.detections.append(detection)
                    last_tracked_objects.append(detection)
                    self.stats['total_detections'] += 1
            
            # Visualization
            if writer:
                frame_vis = frame.copy()
                for track_id, (cx, cy) in tracked.items():
                    cx, cy = int(cx), int(cy)
                    
                    cv2.circle(frame_vis, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame_vis, f"ID:{track_id}", 
                              (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                writer.write(frame_vis)
            
            # Progress
            self.stats['frames_processed'] += 1
            frame_id += 1
            
            if frame_id % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
                unique_objs = len(set([d['object_id'] for d in self.detections]))
                print(f"⏳ Frame {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) | "
                      f"FPS: {current_fps:.1f} | Objects: {unique_objs}")
        
        cap.release()
        if writer:
            writer.release()
        
        total_time = time.time() - start_time
        
        # Save results
        if self.detections:
            df = pd.DataFrame(self.detections)
            df = df.sort_values(['frame_id', 'object_id'])
            Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_csv, index=False)
            
            unique_objects = df['object_id'].nunique()
            
            print(f"\n💾 Saved {len(df)} detections to {output_csv}")
            
            print("\n" + "="*60)
            print("✅ PROCESSING COMPLETE")
            print("="*60)
            print(f"📊 Statistics:")
            print(f"   • Frames processed: {self.stats['frames_processed']}/{total_frames}")
            print(f"   • Frames skipped: {self.stats['frames_skipped']}")
            print(f"   • Total time: {total_time:.1f}s")
            print(f"   • Average FPS: {self.stats['frames_processed']/total_time:.1f}")
            print(f"   • Inference time: {self.stats['inference_time']:.1f}s ({self.stats['inference_time']/total_time*100:.1f}%)")
            print(f"   • Tracking time: {self.stats['tracking_time']:.1f}s ({self.stats['tracking_time']/total_time*100:.1f}%)")
            print(f"   • Detections: {self.stats['total_detections']}")
            print(f"   • Unique objects: {unique_objects}")
            
            if self.skip_frames > 1:
                speedup = (total_frames / self.stats['frames_processed'])
                time_saved = total_time * (speedup - 1) / speedup
                print(f"\n🚀 Optimization:")
                print(f"   • Skip factor: {self.skip_frames}x")
                print(f"   • Speedup: {speedup:.2f}x")
                print(f"   • Time saved: ~{time_saved:.1f}s")
            
            print("="*60)
        else:
            print("⚠️ No detections to save")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Optimized Problem 1: Detection + Tracking')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='submissions/p1_detection_obb_optimized.csv',
                       help='Output CSV file')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 2=skip 50%%, etc.)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated video')
    parser.add_argument('--output-video', type=str,
                       help='Output video path')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable performance optimizations')
    
    args = parser.parse_args()
    
    # Create detector
    detector = OptimizedDroneDetector(
        model_path='yolov8n-obb.pt',
        conf_threshold=args.conf,
        device='auto',
        optimize=not args.no_optimize,
        skip_frames=args.skip
    )
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_csv=args.output,
        save_video=args.save_video,
        output_video=args.output_video
    )


if __name__ == '__main__':
    main()
