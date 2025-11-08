"""
TESA Defence - Problem 1: Object Detection & Tracking
11 พ.ย. 2025

Output Format: frame_id, object_id, center_x, center_y, w, h, theta (normalized 0-1 except theta in degrees)
"""

import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from ultralytics import YOLO
import torch
from byte_track_wrapper import ByteTrackWrapper


class DroneDetectionTracker:
    """Drone detection and tracking for competition Problem 1"""
    
    def __init__(self, model_path='yolov8n-obb.pt', device='auto', use_bytetrack=True, video_fps=30):
        """
        Initialize detector and tracker
        
        Args:
            model_path: Path to YOLO-OBB model
            device: 'auto', 'cuda', or 'cpu'
            use_bytetrack: Use ByteTrack instead of CentroidTracker
            video_fps: Video frame rate (will be updated when processing video)
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔧 Device: {self.device}")
        else:
            self.device = device
        
        # Load YOLO model
        print(f"📦 Loading YOLO-OBB: {model_path}")
        self.model = YOLO(model_path)
        
        # Store tracker settings
        self.use_bytetrack = use_bytetrack
        self.video_fps = video_fps
        self.tracker = None  # Will be initialized in process_video with correct FPS
        
        # Detection storage
        self.detections = []
        
    def process_video(self, video_path, output_csv, conf_threshold=0.4, 
                     display=False, save_video=None):
        """
        Process video and export detections
        
        Args:
            video_path: Path to input video
            output_csv: Path to output CSV file
            conf_threshold: Detection confidence threshold
            display: Show video while processing
            save_video: Path to save annotated video (optional)
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracker with correct FPS
        if self.tracker is None:
            if self.use_bytetrack:
                print(f"📍 Tracker: ByteTrack (FPS: {fps})")
                self.tracker = ByteTrackWrapper(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=fps,  # Use video's FPS
                    track_history=30
                )
            else:
                print(f"📍 Tracker: CentroidTracker")
                from centroid_tracker import CentroidTracker
                self.tracker = CentroidTracker(
                    max_disappeared=30,
                    max_distance=100,
                    track_history=30
                )
        
        print(f"\n{'='*60}")
        print(f"📹 Video: {Path(video_path).name}")
        print(f"📐 Resolution: {width}x{height} @ {fps} FPS")
        print(f"🎬 Total frames: {total_frames}")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        print(f"{'='*60}\n")
        
        # Video writer
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
            print(f"💾 Saving video to: {save_video}")
        
        # Processing
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Run YOLO detection
                results = self.model(
                    frame,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Extract detections
                centroids = []
                bboxes = []
                
                if results[0].obb is not None:
                    for obb in results[0].obb:
                        # Get OBB parameters
                        xywhr = obb.xywhr[0].cpu().numpy()
                        cx, cy = int(xywhr[0]), int(xywhr[1])
                        w, h = int(xywhr[2]), int(xywhr[3])
                        theta = np.degrees(xywhr[4])
                        conf = float(obb.conf[0])
                        
                        # Store centroid and bbox
                        centroids.append((cx, cy))
                        bboxes.append({
                            'cx': cx,
                            'cy': cy,
                            'w': w,
                            'h': h,
                            'theta': theta,
                            'conf': conf,
                            'points': obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        })
                
                # Update tracker
                objects = self.tracker.update(centroids, current_time)
                
                # Log detections with bounding boxes
                self._log_detections(frame_count, objects, bboxes, width, height)
                
                # Draw annotations
                annotated = self._draw_frame(frame, objects, bboxes, frame_count)
                
                # Display
                if display:
                    cv2.imshow('Problem 1: Detection & Tracking', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️ Stopped by user")
                        break
                
                # Save frame
                if writer:
                    writer.write(annotated)
                
                # Progress
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps_actual:.1f} | Objects: {len(objects)}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted (Ctrl+C)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Save CSV
            self._save_csv(output_csv)
            
            # Stats
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✅ PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"📊 Statistics:")
            print(f"   • Frames: {frame_count}")
            print(f"   • Time: {elapsed:.1f}s")
            print(f"   • Avg FPS: {frame_count/elapsed:.1f}")
            print(f"   • Detections: {len(self.detections)}")
            print(f"   • Unique objects: {self.tracker.get_total_tracked_objects()}")
            print(f"   • Output: {output_csv}")
            print(f"{'='*60}\n")
    
    def _log_detections(self, frame_num, objects, bboxes, width, height):
        """
        Log detections in competition format (normalized 0-1)
        
        Args:
            frame_num: Frame number
            objects: Tracked objects {id: (cx, cy)}
            bboxes: Detection bounding boxes
            width: Frame width for normalization
            height: Frame height for normalization
        """
        # Match tracked objects to detections
        for object_id, centroid in objects.items():
            # Find closest bbox to this centroid
            min_dist = float('inf')
            matched_bbox = None
            
            for bbox in bboxes:
                dist = np.sqrt((bbox['cx'] - centroid[0])**2 + 
                              (bbox['cy'] - centroid[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    matched_bbox = bbox
            
            if matched_bbox and min_dist < 50:  # Match within 50 pixels
                # Normalize coordinates (0-1)
                center_x = matched_bbox['cx'] / width
                center_y = matched_bbox['cy'] / height
                w_norm = matched_bbox['w'] / width
                h_norm = matched_bbox['h'] / height
                theta = matched_bbox['theta']  # Already in degrees
                
                self.detections.append({
                    'frame_id': frame_num,
                    'object_id': object_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'w': w_norm,
                    'h': h_norm,
                    'theta': round(theta, 1)
                })
    
    def _draw_frame(self, frame, objects, bboxes, frame_num):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Draw all detections
        for bbox in bboxes:
            # Draw OBB polygon
            points = bbox['points']
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(annotated, (bbox['cx'], bbox['cy']), 5, (0, 255, 0), -1)
        
        # Draw tracked objects
        for object_id, centroid in objects.items():
            # Draw ID
            text = f"ID: {object_id}"
            cv2.putText(annotated, text, (centroid[0] - 20, centroid[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw track path
            path = self.tracker.get_track_path(object_id)
            if len(path) > 1:
                for i in range(1, len(path)):
                    cv2.line(annotated, path[i-1], path[i], (255, 0, 255), 2)
        
        # Draw info
        cv2.putText(annotated, f'Frame: {frame_num}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f'Objects: {len(objects)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated
    
    def _save_csv(self, output_path):
        """Save detections to CSV"""
        if not self.detections:
            print("⚠️ No detections to save")
            return
        
        df = pd.DataFrame(self.detections)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(df)} detections to {output_path}")


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Problem 1: Detection & Tracking')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='submissions/problem1_output.csv', 
                       help='Output CSV path')
    parser.add_argument('--conf', type=float, default=0.55, 
                       help='Confidence threshold (default: 0.55)')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='YOLO model path')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save annotated video')
    parser.add_argument('--no-bytetrack', action='store_true',
                       help='Use CentroidTracker instead of ByteTrack')
    
    args = parser.parse_args()
    
    # Create detector
    detector = DroneDetectionTracker(
        model_path=args.model,
        use_bytetrack=not args.no_bytetrack
    )
    
    # Process video
    detector.process_video(
        video_path=args.video,
        output_csv=args.output,
        conf_threshold=args.conf,
        display=args.display,
        save_video=args.save_video
    )
