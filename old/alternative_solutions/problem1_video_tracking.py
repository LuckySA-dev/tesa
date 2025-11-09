"""
Problem 1: Drone Detection + Tracking from Video
=================================================

Complete system for drone detection and tracking with the following features:

Image Processing (OpenCV):
- Video file or webcam input
- Frame-by-frame processing
- FPS calculation and display
- Morphological operations (available in utils)

Deep Learning Model:
- YOLO-OBB (Oriented Bounding Box) detection
- Support for multiple drone types
- Configurable confidence threshold

Object Tracking:
- Centroid tracking with unique IDs
- Euclidean distance matching
- Handle disappeared objects
- Tracking path visualization

Metrics & Visualization:
- Bounding box detection (OBB)
- Path drawing with color coding
- Velocity and direction calculation
- Real-time statistics overlay
- Data logging to CSV

Output:
- Annotated video with tracking
- CSV log with frame-by-frame data
- Real-time display with statistics

Author: TESA Defence Team
Date: November 8, 2025
"""

from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict
import argparse
import torch

from centroid_tracker import CentroidTracker
from api_client import MockSatelliteAPI


class DroneVideoTracker:
    """Complete drone detection and tracking system"""
    
    def __init__(self, model_path='yolov8n-obb.pt', device='auto', enable_api=False):
        """
        Initialize Drone Video Tracker
        
        Args:
            model_path (str): Path to YOLO-OBB model
            device (str): 'auto', 'cuda', or 'cpu'
            enable_api (bool): Enable API communication to satellite
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔧 Auto-detected device: {self.device}")
        else:
            self.device = device
        
        print(f"📦 Loading YOLO-OBB model: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize centroid tracker
        self.tracker = CentroidTracker(
            max_disappeared=30,   # Keep tracking for 30 frames
            max_distance=100,     # Max 100 pixels to match
            track_history=30      # Keep 30 frames of path
        )
        
        # Initialize API client (Mock for testing)
        self.enable_api = enable_api
        self.api = MockSatelliteAPI() if enable_api else None
        if enable_api:
            print(f"📡 API enabled: Mock Satellite API")
        
        # Colors for visualization (BGR format)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
        ]
        
        # Data logging
        self.log_data = []
        
        # Statistics
        self.total_frames_processed = 0
        self.total_detections = 0
    
    def process_video(self, video_path, output_path=None, output_csv=None,
                     conf_threshold=0.25, display=True, save_interval=30):
        """
        Process video with detection and tracking
        
        Args:
            video_path (str|int): Path to video file or 0 for webcam
            output_path (str): Save annotated video (optional)
            output_csv (str): Save tracking log (optional)
            conf_threshold (float): Detection confidence threshold (0.0-1.0)
            display (bool): Show video while processing
            save_interval (int): Save log every N frames
        """
        # Open video capture
        if video_path == 0:
            cap = cv2.VideoCapture(0)
            print("📹 Using webcam...")
        else:
            cap = cv2.VideoCapture(str(video_path))
            print(f"📹 Processing video: {video_path}")
        
        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:  # Webcam or stream
            total_frames = float('inf')
            print(f"📐 Video: {width}x{height} @ {fps} FPS (live stream)")
        else:
            print(f"📐 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output video: {output_path}")
        
        if output_csv:
            print(f"💾 Output log: {output_csv}")
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        
        print("\n" + "="*60)
        print("🎬 Starting video processing...")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save screenshot")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Run YOLO-OBB detection
                results = self.model(
                    frame,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Extract centroids and detection info from OBB results
                centroids = []
                detections_info = []
                
                if results[0].obb is not None:
                    for obb in results[0].obb:
                        # Get OBB parameters
                        xywhr = obb.xywhr[0].cpu().numpy()
                        cx, cy = int(xywhr[0]), int(xywhr[1])
                        w, h = xywhr[2], xywhr[3]
                        theta = np.degrees(xywhr[4])
                        conf = float(obb.conf[0])
                        
                        centroids.append((cx, cy))
                        detections_info.append({
                            'center': (cx, cy),
                            'size': (w, h),
                            'angle': theta,
                            'conf': conf,
                            'points': obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        })
                        
                        self.total_detections += 1
                
                # Update centroid tracker
                objects = self.tracker.update(centroids, current_time)
                
                # API Integration: Send first alarm when drones first detected
                if self.enable_api and len(objects) > 0 and not self.api.first_alarm_sent:
                    if self.api.send_first_alarm(len(objects), frame):
                        print(f"📡 First alarm sent: {len(objects)} drones detected")
                
                # Draw all annotations
                annotated = self._draw_frame(
                    frame, objects, detections_info, frame_count
                )
                
                # Calculate processing FPS
                elapsed = time.time() - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw statistics overlay
                self._draw_stats(annotated, fps_display, len(objects), frame_count, total_frames)
                
                # Log frame data
                self._log_frame_data(frame_count, objects, current_time, fps)
                
                # Display
                if display:
                    cv2.imshow('TESA Defence - Drone Tracking System', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n⏹️  Stopping by user request...")
                        break
                    elif key == ord('p'):
                        print("⏸️  Paused. Press any key to continue...")
                        cv2.waitKey(0)
                    elif key == ord('s'):
                        screenshot_path = f'screenshot_frame_{frame_count:06d}.jpg'
                        cv2.imwrite(screenshot_path, annotated)
                        print(f"📸 Screenshot saved: {screenshot_path}")
                
                # Save video frame
                if writer:
                    writer.write(annotated)
                
                # Progress update every 30 frames
                if frame_count % 30 == 0 and total_frames != float('inf'):
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps_display:.1f} | Drones: {len(objects)}")
                
                # Save log periodically
                if output_csv and frame_count % save_interval == 0:
                    self._save_log(output_csv)
                
                # API Integration: Send tracking data periodically (every 60 frames ≈ 2 seconds @ 30fps)
                if self.enable_api and frame_count % 60 == 0 and len(objects) > 0:
                    # Build tracking data for all objects
                    tracking_objects = []
                    for objectID, centroid in objects.items():
                        # Mock GPS conversion
                        lat = 13.7563 + (centroid[1] - 682) / 10000
                        lon = 100.5018 + (centroid[0] - 1024) / 10000
                        
                        # Mock drone type
                        drone_types = ['DJI_Mavic', 'DJI_Phantom', 'Generic_Drone', 'Racing_Drone']
                        drone_type = drone_types[(objectID - 1) % len(drone_types)]
                        
                        velocity = self.tracker.calculate_velocity(objectID, current_time, fps)
                        tracking_objects.append({
                            'frame': frame_count,
                            'object_id': objectID,
                            'drone_type': drone_type,
                            'lat': lat,
                            'lon': lon,
                            'speed_ms': velocity['speed'] if velocity else 0,
                            'direction_deg': velocity['direction'] if velocity else 0
                        })
                    
                    # Send all objects at once
                    if self.api.send_tracking_data(tracking_objects, frame, include_image=True):
                        print(f"📡 Tracking data sent: {len(tracking_objects)} drones @ frame {frame_count}")
                
                self.total_frames_processed = frame_count
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user (Ctrl+C)")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Save final log
            if output_csv and len(self.log_data) > 0:
                self._save_log(output_csv)
            
            # Print summary
            self._print_summary(fps_display, output_path, output_csv)
    
    def _draw_frame(self, frame, objects, detections_info, frame_count):
        """
        Draw all annotations on frame
        
        Args:
            frame: Input frame
            objects: Tracked objects from centroid tracker
            detections_info: Detection information from YOLO
            frame_count: Current frame number
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # 1. Draw OBB bounding boxes
        for det in detections_info:
            points = det['points']
            conf = det['conf']
            
            # Draw rotated rectangle (OBB)
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
            
            # Draw confidence score
            label = f'{conf:.2f}'
            cv2.putText(annotated, label,
                       (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 2. Draw tracked objects with IDs and paths
        for objectID, centroid in objects.items():
            # Get color for this ID
            color = self.colors[objectID % len(self.colors)]
            
            # Draw tracking path (with fade effect)
            path = self.tracker.get_track_path(objectID)
            if len(path) > 1:
                for i in range(1, len(path)):
                    # Fade older points
                    thickness = max(1, int(3 * (i / len(path))))
                    cv2.line(annotated, path[i-1], path[i], color, thickness)
            
            # Draw centroid point
            cv2.circle(annotated, centroid, 5, color, -1)
            cv2.circle(annotated, centroid, 20, color, 2)
            
            # Draw object ID
            text = f"ID {objectID}"
            cv2.putText(annotated, text,
                       (centroid[0] - 30, centroid[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw velocity and direction
            velocity = self.tracker.calculate_velocity(objectID, time.time())
            if velocity['speed'] > 0.1:  # Only show if moving
                vel_text = f"{velocity['speed']:.1f}m/s {velocity['direction']:.0f}°"
                cv2.putText(annotated, vel_text,
                           (centroid[0] - 40, centroid[1] + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def _draw_stats(self, frame, fps, drone_count, frame_num, total_frames):
        """
        Draw statistics overlay
        
        Args:
            frame: Frame to draw on
            fps: Current FPS
            drone_count: Number of tracked drones
            frame_num: Current frame number
            total_frames: Total frames in video
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (380, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Draw text statistics
        y_offset = 30
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        # Active drones
        cv2.putText(frame, f'Active Drones: {drone_count}', (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_offset += 35
        # Total tracked
        cv2.putText(frame, f'Total Tracked: {self.tracker.get_total_tracked_objects()}', 
                   (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        y_offset += 35
        # Frame counter
        if total_frames != float('inf'):
            progress = f'{frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)'
        else:
            progress = f'{frame_num}'
        cv2.putText(frame, f'Frame: {progress}', (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        # System label
        cv2.putText(frame, 'TESA Defence System', (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    def _log_frame_data(self, frame_num, objects, timestamp, fps):
        """
        Log tracking data for current frame
        
        Args:
            frame_num: Frame number
            objects: Tracked objects
            timestamp: Current timestamp
            fps: Video FPS
        """
        for objectID, centroid in objects.items():
            velocity = self.tracker.calculate_velocity(objectID, timestamp, fps)
            
            # Mock GPS conversion (TODO: implement real conversion)
            # For now, use simple offset from center
            lat = 13.7563 + (centroid[1] - 682) / 10000  # Mock latitude
            lon = 100.5018 + (centroid[0] - 1024) / 10000  # Mock longitude
            
            # Mock drone type (TODO: implement real classification)
            # For now, cycle through types based on ID
            drone_types = ['DJI_Mavic', 'DJI_Phantom', 'Generic_Drone', 'Racing_Drone']
            drone_type = drone_types[(objectID - 1) % len(drone_types)]
            
            self.log_data.append({
                'frame': frame_num,
                'timestamp': round(timestamp, 3),
                'object_id': objectID,
                'drone_type': drone_type,  # Added
                'center_x': centroid[0],
                'center_y': centroid[1],
                'lat': round(lat, 6),  # Added
                'lon': round(lon, 6),  # Added
                'speed_ms': velocity['speed'],
                'direction_deg': velocity['direction'],
                'distance_pixels': velocity.get('distance_pixels', 0),
                'confidence': 0.85  # Mock confidence (TODO: get from detection)
            })
    
    def _save_log(self, output_csv):
        """Save tracking log to CSV"""
        if len(self.log_data) > 0:
            df = pd.DataFrame(self.log_data)
            df.to_csv(output_csv, index=False)
            print(f"💾 Log saved: {output_csv} ({len(self.log_data)} records)")
    
    def _print_summary(self, fps, output_path, output_csv):
        """Print processing summary"""
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   • Frames processed: {self.total_frames_processed}")
        print(f"   • Average FPS: {fps:.1f}")
        print(f"   • Total detections: {self.total_detections}")
        print(f"   • Unique drones tracked: {self.tracker.get_total_tracked_objects()}")
        
        if output_path:
            print(f"\n📹 Output:")
            print(f"   • Video: {output_path}")
        
        if output_csv and len(self.log_data) > 0:
            print(f"   • Log: {output_csv} ({len(self.log_data)} records)")
        
        print("="*60 + "\n")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='TESA Defence - Problem 1: Drone Video Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with output
  python problem1_video_tracking.py --video videos/test.mp4 --output result.mp4
  
  # Use webcam (real-time)
  python problem1_video_tracking.py --video 0
  
  # With custom model and confidence
  python problem1_video_tracking.py --video test.mp4 --model yolov8s-obb.pt --conf 0.3
  
  # Process without display (headless)
  python problem1_video_tracking.py --video test.mp4 --output result.mp4 --no-display

For more information, visit: https://github.com/tesa-defence
        """
    )
    
    parser.add_argument('--video', type=str, default='0',
                       help='Video file path or 0 for webcam (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--log', type=str, default='p1_tracking_log.csv',
                       help='Output CSV log path (default: p1_tracking_log.csv)')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='YOLO-OBB model path (default: yolov8n-obb.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Detection confidence threshold 0.0-1.0 (default: 0.25)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (for headless systems)')
    parser.add_argument('--save-interval', type=int, default=30,
                       help='Save log every N frames (default: 30)')
    
    args = parser.parse_args()
    
    # Convert video argument
    video_path = 0 if args.video == '0' else args.video
    
    # Validate video file exists (if not webcam)
    if video_path != 0 and not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Create tracker
    print("\n" + "="*60)
    print("🚀 TESA DEFENCE - DRONE TRACKING SYSTEM")
    print("="*60)
    
    tracker = DroneVideoTracker(model_path=args.model, device=args.device)
    
    # Process video
    tracker.process_video(
        video_path=video_path,
        output_path=args.output,
        output_csv=args.log,
        conf_threshold=args.conf,
        display=not args.no_display,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
