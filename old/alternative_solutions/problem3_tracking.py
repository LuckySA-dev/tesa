"""
Problem 3: Drone Tracking
==========================

ติดตามโดรนในวิดีโอและรักษา track_id ให้คงที่ตลอดคลิป
Output: p3_tracking_obb.csv

Format: video_id, frame_id, track_id, center_x, center_y, w, h, theta
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from ultralytics import YOLO
from collections import defaultdict
import torch


class ByteTracker:
    """
    Simplified ByteTrack algorithm for drone tracking
    
    ByteTrack ใช้ IoU matching และ Kalman filter เพื่อติดตามวัตถุ
    """
    
    def __init__(self, 
                 track_thresh=0.5,
                 track_buffer=30,
                 match_thresh=0.8,
                 min_track_len=10):
        """
        Args:
            track_thresh: threshold สำหรับ high score detection
            track_buffer: จำนวน frame ที่เก็บ track หาย
            match_thresh: threshold สำหรับ matching IoU
            min_track_len: จำนวน frame ขั้นต่ำที่ track ต้องมีก่อนจะนับ
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_track_len = min_track_len
        
        self.tracked_objects = []
        self.lost_tracks = []
        self.removed_tracks = []
        
        self.frame_id = 0
        self.track_id_count = 1  # เริ่มจาก 1 ตามโจทย์ (1,2,3,...)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks ด้วย detection ใหม่
        
        Args:
            detections: list of dict with keys: center_x, center_y, w, h, theta, conf
            
        Returns:
            list of tracked objects with track_id
        """
        self.frame_id += 1
        
        # แยก detection ตาม confidence
        high_conf_dets = [d for d in detections if d.get('conf', 0) >= self.track_thresh]
        low_conf_dets = [d for d in detections if d.get('conf', 0) < self.track_thresh]
        
        # Match กับ tracks ที่มีอยู่
        matched_tracks, unmatched_tracks, unmatched_dets = self._match(
            self.tracked_objects, high_conf_dets
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracked_objects[track_idx].update(high_conf_dets[det_idx])
        
        # สร้าง tracks ใหม่จาก unmatched high confidence detections
        for det_idx in unmatched_dets:
            new_track = Track(high_conf_dets[det_idx], self.track_id_count)
            self.track_id_count += 1
            self.tracked_objects.append(new_track)
        
        # ลอง match lost tracks กับ low confidence detections
        if len(low_conf_dets) > 0 and len(unmatched_tracks) > 0:
            lost_tracks = [self.tracked_objects[i] for i in unmatched_tracks]
            matched_lost, _, _ = self._match(lost_tracks, low_conf_dets)
            
            for track_idx, det_idx in matched_lost:
                lost_tracks[track_idx].update(low_conf_dets[det_idx])
        
        # ลบ tracks ที่หายไปนานเกินไป
        self.tracked_objects = [
            t for t in self.tracked_objects 
            if t.frames_lost <= self.track_buffer
        ]
        
        # Return active tracks (no filtering here, will filter later)
        return [t.to_dict() for t in self.tracked_objects if t.frames_lost == 0]
    
    def get_confirmed_tracks(self) -> List[Dict]:
        """Get only tracks that meet minimum length requirement"""
        return [
            t.to_dict() for t in self.tracked_objects 
            if t.frames_lost == 0 and t.age >= self.min_track_len
        ]
    
    def _match(self, tracks: List, detections: List) -> Tuple[List, List, List]:
        """
        Match tracks กับ detections ด้วย IoU
        
        Returns:
            (matched_pairs, unmatched_track_indices, unmatched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # คำนวณ IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou_obb(track.current_state, det)
        
        # Greedy matching
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # หา max IoU
            max_iou = 0
            max_i, max_j = -1, -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j
            
            if max_iou >= self.match_thresh:
                matched.append((max_i, max_j))
                unmatched_tracks.remove(max_i)
                unmatched_dets.remove(max_j)
            else:
                break
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou_obb(self, obb1: Dict, obb2: Dict) -> float:
        """
        คำนวณ IoU ระหว่าง 2 OBB (simplified)
        
        Note: การคำนวณ IoU ของ OBB ที่แม่นต้องซับซ้อน
        ที่นี่ใช้วิธีประมาณด้วย bounding box
        """
        # แปลงเป็น axis-aligned bounding box
        def obb_to_bbox(obb):
            cx, cy, w, h = obb['center_x'], obb['center_y'], obb['w'], obb['h']
            # ประมาณด้วย bbox ที่ครอบ OBB
            half_diag = np.sqrt(w**2 + h**2) / 2
            return {
                'x1': cx - half_diag,
                'y1': cy - half_diag,
                'x2': cx + half_diag,
                'y2': cy + half_diag
            }
        
        box1 = obb_to_bbox(obb1)
        box2 = obb_to_bbox(obb2)
        
        # คำนวณ IoU
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


class Track:
    """Class สำหรับเก็บข้อมูล track แต่ละ object"""
    
    def __init__(self, detection: Dict, track_id: int):
        self.track_id = track_id
        self.current_state = detection
        self.frames_lost = 0
        self.age = 1
    
    def update(self, detection: Dict):
        """Update track ด้วย detection ใหม่"""
        self.current_state = detection
        self.frames_lost = 0
        self.age += 1
    
    def mark_lost(self):
        """ทำเครื่องหมายว่า track หาย"""
        self.frames_lost += 1
    
    def to_dict(self) -> Dict:
        """แปลงเป็น dict พร้อม track_id"""
        return {
            'track_id': self.track_id,
            **self.current_state
        }


class DroneTracker:
    """ระบบติดตามโดรนในวิดีโอ"""
    
    def __init__(self, model_path='yolov8m-obb.pt', device='auto'):
        """
        Args:
            model_path: YOLO-OBB model path
            device: 'auto', 'cuda', or 'cpu'
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Auto-detected device: {self.device}")
        else:
            self.device = device
        
        print(f"Loading YOLO-OBB model: {model_path}")
        self.model = YOLO(model_path)
    
    def process_video(self,
                     video_path: str,
                     video_id: str,
                     output_csv: str = None,
                     conf_threshold: float = 0.25,
                     track_thresh: float = 0.5,
                     skip_frames: int = 0,
                     match_thresh: float = 0.6) -> pd.DataFrame:
        """
        ประมวลผลวิดีโอและติดตามโดรน
        
        Args:
            video_path: path to video file
            video_id: รหัสวิดีโอ
            output_csv: output file (ถ้าเป็น None จะไม่บันทึก)
            conf_threshold: confidence threshold สำหรับ detection
            track_thresh: threshold สำหรับ tracking
            skip_frames: skip ทุกๆ n frames (0 = ประมวลผลทุก frame)
            
        Returns:
            DataFrame with tracking results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {video_path}")
        print(f"  - Frames: {total_frames}, FPS: {fps:.2f}")
        print(f"  - Resolution: {width}x{height}")
        
        # สร้าง tracker
        tracker = ByteTracker(track_thresh=track_thresh, match_thresh=match_thresh)
        
        results = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames ถ้าต้องการ
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Run detection
            detections = self._detect_frame(frame, conf_threshold)
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # บันทึกผลลัพธ์
            for track in tracks:
                results.append({
                    'video_id': video_id,
                    'frame_id': frame_count,
                    'track_id': track['track_id'],
                    'center_x': round(track['center_x'], 6),
                    'center_y': round(track['center_y'], 6),
                    'w': round(track['w'], 6),
                    'h': round(track['h'], 6),
                    'theta': round(track['theta'], 2)
                })
            
            processed_count += 1
            if processed_count % 30 == 0:
                print(f"  Processed: {frame_count}/{total_frames} frames ({len(tracks)} tracks)", end='\r')
            
            frame_count += 1
        
        cap.release()
        print(f"\n  ✓ Complete: {processed_count} frames processed")
        
        # สร้าง DataFrame
        df = pd.DataFrame(results)
        
        # Filter out short tracks (spurious detections) - เพิ่มเป็น 30 frames
        if len(df) > 0:
            track_lengths = df.groupby('track_id').size()
            min_len = max(30, tracker.min_track_len)  # ใช้อย่างน้อย 30 frames
            valid_tracks = track_lengths[track_lengths >= min_len].index
            
            df_filtered = df[df['track_id'].isin(valid_tracks)].copy()
            
            n_removed = len(df) - len(df_filtered)
            n_tracks_removed = len(track_lengths) - len(valid_tracks)
            
            print(f"  ✓ Track filtering: Removed {n_tracks_removed} short tracks ({n_removed} detections)")
            df = df_filtered
        
        if output_csv and len(df) > 0:
            df.to_csv(output_csv, index=False)
            print(f"  ✓ Saved to: {output_csv}")
        
        return df
    
    def _detect_frame(self, frame: np.ndarray, conf_threshold: float, 
                     iou_threshold: float = 0.7) -> List[Dict]:
        """ตรวจจับโดรนใน frame เดียว พร้อม duplicate removal"""
        img_h, img_w = frame.shape[:2]
        
        # Run YOLO with higher IoU threshold for better NMS
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, 
                           device=self.device, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].obb is not None:
            for obb in results[0].obb:
                xywhr = obb.xywhr[0].cpu().numpy()
                conf = float(obb.conf[0].cpu().numpy())
                
                # Normalize
                center_x = float(xywhr[0] / img_w)
                center_y = float(xywhr[1] / img_h)
                w = float(xywhr[2] / img_w)
                h = float(xywhr[3] / img_h)
                theta = float(np.degrees(xywhr[4]))
                
                # Normalize theta
                while theta > 90:
                    theta -= 180
                while theta < -90:
                    theta += 180
                
                # Use smallest angle representation
                if theta > 45:
                    theta -= 90
                elif theta < -45:
                    theta += 90
                
                # Filter out detections near image edges (likely false positives)
                # Skip if too close to bottom edge (y > 0.85) or too close to left/right edge
                if center_y > 0.85 or center_x < 0.05:
                    continue
                
                detections.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'w': w,
                    'h': h,
                    'theta': theta,
                    'conf': conf
                })
        
        # Post-process: Remove duplicates (เข้มงวดขึ้นเป็น 0.4)
        original_count = len(detections)
        detections = self._remove_duplicates(detections, iou_threshold=0.4)
        
        if original_count > 0 and len(detections) == 0:
            print(f"\nWARNING: Duplicate removal removed ALL {original_count} detections!")
        
        return detections
    
    def _remove_duplicates(self, detections: List[Dict], iou_threshold=0.3) -> List[Dict]:
        """ลบ duplicate detections โดยใช้ IoU"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
        
        keep = []
        for det in detections:
            # ตรวจสอบว่า overlap กับ detections ที่เก็บไว้แล้วหรือไม่
            is_duplicate = False
            for kept_det in keep:
                iou = self._compute_iou_simple(det, kept_det)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(det)
        
        # DON'T remove conf field - ByteTracker needs it!
        return keep
    
    def _compute_iou_simple(self, det1: Dict, det2: Dict) -> float:
        """คำนวณ IoU แบบง่าย (approximate with bounding box)"""
        def get_bbox(det):
            cx, cy, w, h = det['center_x'], det['center_y'], det['w'], det['h']
            diag = np.sqrt(w**2 + h**2)
            return {
                'x1': cx - diag/2, 'y1': cy - diag/2,
                'x2': cx + diag/2, 'y2': cy + diag/2
            }
        
        box1 = get_bbox(det1)
        box2 = get_bbox(det2)
        
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
    
    def process_multiple_videos(self,
                               video_folder: str,
                               output_csv: str = 'p3_tracking_obb.csv',
                               **kwargs) -> pd.DataFrame:
        """ประมวลผลหลายวิดีโอ"""
        video_folder = Path(video_folder)
        video_files = list(video_folder.glob('*.mp4')) + \
                     list(video_folder.glob('*.avi')) + \
                     list(video_folder.glob('*.mov'))
        
        print(f"Found {len(video_files)} videos in {video_folder}")
        
        all_results = []
        
        for i, video_path in enumerate(sorted(video_files), 1):
            video_id = video_path.stem  # ใช้ชื่อไฟล์เป็น video_id
            print(f"\n[{i}/{len(video_files)}] Processing: {video_id}")
            
            df = self.process_video(video_path, video_id, output_csv=None, **kwargs)
            all_results.append(df)
        
        # รวมผลลัพธ์
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ All videos processed!")
        print(f"  - Total videos: {len(video_files)}")
        print(f"  - Total tracks: {len(df_all)}")
        print(f"  - Output: {output_csv}")
        print(f"{'='*60}")
        
        return df_all


def main():
    parser = argparse.ArgumentParser(description='Problem 3: Drone Tracking')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--videos', type=str, help='Path to folder with videos')
    parser.add_argument('--output', type=str, default='p3_tracking_obb.csv',
                       help='Output CSV file')
    parser.add_argument('--model', type=str, default='yolov8m-obb.pt',
                       help='YOLO-OBB model path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--track-thresh', type=float, default=0.5,
                       help='Tracking threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip every N frames (0=process all)')
    parser.add_argument('--match-thresh', type=float, default=0.6,
                       help='IoU threshold for track matching (lower = more tolerant)')
    
    args = parser.parse_args()
    
    tracker = DroneTracker(model_path=args.model, device=args.device)
    
    if args.video:
        # ประมวลผลวิดีโอเดียว
        video_id = Path(args.video).stem
        df = tracker.process_video(
            video_path=args.video,
            video_id=video_id,
            output_csv=args.output,
            conf_threshold=args.conf,
            track_thresh=args.track_thresh,
            skip_frames=args.skip_frames,
            match_thresh=args.match_thresh
        )
    elif args.videos:
        # ประมวลผลหลายวิดีโอ
        df = tracker.process_multiple_videos(
            video_folder=args.videos,
            output_csv=args.output,
            conf_threshold=args.conf,
            track_thresh=args.track_thresh,
            skip_frames=args.skip_frames,
            match_thresh=args.match_thresh
        )
    else:
        print("Error: Please specify --video or --videos")
        return
    
    print(f"\nSample results:")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
