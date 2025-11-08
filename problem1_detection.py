"""
Problem 1: Drone Detection with OBB (Oriented Bounding Box)
============================================================

Image-based drone detection system using YOLO-OBB (Oriented Bounding Box).
This module processes single images or batches of images to detect drones
with rotated bounding boxes.

Features:
- YOLO-OBB detection (supports rotated objects)
- Batch image processing
- CSV export in YOLO-OBB format
- Duplicate removal
- Configurable confidence threshold

Output Format (p1_detection_obb.csv):
- img_file: Image filename
- center_x, center_y: Normalized center coordinates (0-1)
- w, h: Normalized width and height (0-1)
- theta: Rotation angle in degrees (-90 to +90)

Note: For video processing and tracking, use problem1_video_tracking.py instead.

Author: TESA Defence Team
Date: November 8, 2025
"""

from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict
import argparse
import torch


class DroneDetectorOBB:
    """
    Drone Detection with YOLO-OBB (Oriented Bounding Box)
    
    Detects drones in images using YOLO-OBB model and outputs detection
    results in normalized YOLO-OBB format.
    """
    
    def __init__(self, model_path='yolov8n-obb.pt', device='auto'):
        """
        Initialize Drone Detector with YOLO-OBB model
        
        Args:
            model_path (str): Path to YOLO-OBB model file
                Recommended models:
                - 'yolov8n-obb.pt' (nano) - Fast, for Raspberry Pi 5
                - 'yolov8s-obb.pt' (small) - Balanced
                - 'yolov8m-obb.pt' (medium) - Better accuracy, development
                - 'yolov8l-obb.pt' (large) - High accuracy, slow
                - 'yolov8x-obb.pt' (xlarge) - Best accuracy, very slow
            device (str): Device to use
                - 'auto': Auto-detect CUDA/CPU
                - 'cuda': Force GPU
                - 'cpu': Force CPU
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔧 Auto-detected device: {self.device}")
        else:
            self.device = device
        
        print(f"📦 Loading YOLO-OBB model: {model_path}")
        self.model = YOLO(model_path)
        
    def detect_single_image(self, image_path: str, conf_threshold=0.25, 
                          iou_threshold=0.45) -> List[Dict]:
        """
        ตรวจจับโดรนในภาพเดียว
        
        Args:
            image_path: path to image file
            conf_threshold: confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            list of dict: [{'center_x': 0.5, 'center_y': 0.4, 'w': 0.2, 'h': 0.15, 'theta': 10.0}, ...]
        """
        # Run detection
        results = self.model(
            image_path, 
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].obb is not None:
            # อ่านภาพเพื่อหาขนาด
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Cannot read image {image_path}")
                return detections
                
            img_h, img_w = img.shape[:2]
            
            # ดึงข้อมูล OBB
            for obb in results[0].obb:
                # obb.xywhr = [center_x_pixel, center_y_pixel, width_pixel, height_pixel, rotation_rad]
                xywhr = obb.xywhr[0].cpu().numpy()
                
                # Normalize coordinates to 0-1
                center_x = float(xywhr[0] / img_w)
                center_y = float(xywhr[1] / img_h)
                w = float(xywhr[2] / img_w)
                h = float(xywhr[3] / img_h)
                
                # Convert rotation from radians to degrees
                theta_deg = float(np.degrees(xywhr[4]))
                
                # Normalize theta to [-90, 90] range
                while theta_deg > 90:
                    theta_deg -= 180
                while theta_deg < -90:
                    theta_deg += 180
                
                # Use smallest angle representation (prefer angles closer to 0)
                # ถ้ามุมมากกว่า 45° ให้ลบ 90 เพื่อให้ได้มุมที่เล็กกว่า
                if theta_deg > 45:
                    theta_deg -= 90
                elif theta_deg < -45:
                    theta_deg += 90
                
                # Get confidence score
                conf = float(obb.conf[0].cpu().numpy()) if hasattr(obb, 'conf') else 1.0
                
                detections.append({
                    'center_x': round(center_x, 6),
                    'center_y': round(center_y, 6),
                    'w': round(w, 6),
                    'h': round(h, 6),
                    'theta': round(theta_deg, 2),
                    'conf': conf
                })
        
        # Post-process: Remove duplicates based on IoU
        detections = self._remove_duplicates(detections, iou_threshold=0.3)
        
        return detections
    
    def _remove_duplicates(self, detections: List[Dict], iou_threshold=0.3) -> List[Dict]:
        """
        ลบ duplicate detections โดยใช้ IoU
        เก็บ detection ที่มี confidence สูงกว่า
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            # ตรวจสอบว่า overlap กับ detections ที่เก็บไว้แล้วหรือไม่
            is_duplicate = False
            for kept_det in keep:
                iou = self._compute_iou_simple(det, kept_det)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(det)
        
        # Remove conf field before returning
        for det in keep:
            det.pop('conf', None)
        
        return keep
    
    def _compute_iou_simple(self, det1: Dict, det2: Dict) -> float:
        """คำนวณ IoU แบบง่าย (approximate with bounding box)"""
        # ใช้ bounding box ครอบ OBB
        def get_bbox(det):
            cx, cy, w, h = det['center_x'], det['center_y'], det['w'], det['h']
            # ประมาณด้วย diagonal
            diag = np.sqrt(w**2 + h**2)
            return {
                'x1': cx - diag/2, 'y1': cy - diag/2,
                'x2': cx + diag/2, 'y2': cy + diag/2
            }
        
        box1 = get_bbox(det1)
        box2 = get_bbox(det2)
        
        # Intersection
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_dataset(self, image_folder: str, output_csv='p1_detection_obb.csv',
                       conf_threshold=0.25, iou_threshold=0.45) -> pd.DataFrame:
        """
        Process all images in folder and create detection CSV
        
        Args:
            image_folder (str): Path to folder containing images
            output_csv (str): Output CSV file path
            conf_threshold (float): Detection confidence threshold (0.0-1.0)
            iou_threshold (float): IoU threshold for NMS and duplicate removal
            
        Returns:
            pd.DataFrame: Detection results with columns:
                - img_file: Image filename
                - center_x, center_y: Normalized coordinates (0-1)
                - w, h: Normalized width/height (0-1)
                - theta: Rotation angle in degrees (-90 to +90)
        """
        image_folder = Path(image_folder)
        
        if not image_folder.exists():
            raise FileNotFoundError(f"❌ Image folder not found: {image_folder}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f'*{ext}'))
            image_files.extend(image_folder.glob(f'*{ext.upper()}'))
        
        image_files = sorted(set(image_files))
        print(f"📁 Found {len(image_files)} images in {image_folder}")
        
        if len(image_files) == 0:
            print("⚠️  Warning: No images found!")
            return pd.DataFrame()
        
        # Process each image
        results = []
        total_detections = 0
        
        print("\n" + "="*60)
        print("🚀 Starting batch processing...")
        print("="*60 + "\n")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {img_path.name:<30}", end=' ')
            
            detections = self.detect_single_image(
                img_path, 
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            print(f"→ {len(detections)} drone(s)")
            total_detections += len(detections)
            
            # Add each detection to results
            for det in detections:
                results.append({
                    'img_file': img_path.name,
                    **det
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save CSV
        if len(df) > 0:
            df.to_csv(output_csv, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Summary:")
        print(f"   • Total images: {len(image_files)}")
        print(f"   • Images with drones: {df['img_file'].nunique() if len(df) > 0 else 0}")
        print(f"   • Total detections: {total_detections}")
        print(f"   • Output CSV: {output_csv}")
        print("="*60 + "\n")
        
        return df


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='TESA Defence - Problem 1: Drone Detection with OBB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images in folder with default settings
  python problem1_detection.py --images images/drones
  
  # Use custom model with different threshold
  python problem1_detection.py --images images/drones --model yolov8s-obb.pt --conf 0.3
  
  # Force CPU processing
  python problem1_detection.py --images images/drones --device cpu

Recommended models:
  - yolov8n-obb.pt: Fastest, for Raspberry Pi 5
  - yolov8s-obb.pt: Balanced speed/accuracy
  - yolov8m-obb.pt: Better accuracy (default)
  
For video processing, use: problem1_video_tracking.py
        """
    )
    
    parser.add_argument('--images', type=str, default='images/p1_images',
                       help='Path to image folder (default: images/p1_images)')
    parser.add_argument('--output', type=str, default='p1_detection_obb.csv',
                       help='Output CSV file path (default: p1_detection_obb.csv)')
    parser.add_argument('--model', type=str, default='yolov8n-obb.pt',
                       help='YOLO-OBB model path (default: yolov8n-obb.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold 0.0-1.0 (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu (default: auto)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("🚀 TESA DEFENCE - DRONE DETECTION SYSTEM")
    print("="*60)
    print(f"📦 Model: {args.model}")
    print(f"📁 Input: {args.images}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 Confidence: {args.conf}")
    print("="*60 + "\n")
    
    # Validate input folder
    if not Path(args.images).exists():
        print(f"❌ Error: Input folder not found: {args.images}")
        return
    
    # Create detector
    detector = DroneDetectorOBB(model_path=args.model, device=args.device)
    
    # Process images
    df = detector.process_dataset(
        image_folder=args.images,
        output_csv=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Display sample results
    if len(df) > 0:
        print("📋 Sample detections:")
        print(df.head(10).to_string(index=False))
        print(f"\n... and {max(0, len(df)-10)} more detections\n")
    else:
        print("⚠️  No drones detected in any images\n")
    
    # แสดงตัวอย่างผลลัพธ์
    if len(df) > 0:
        print("Sample results (first 10 rows):")
        print(df.head(10).to_string(index=False))
    else:
        print("No drones detected in any image.")


if __name__ == '__main__':
    main()
