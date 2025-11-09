"""
Problem 2: Drone Localization
==============================

คำนวณตำแหน่งโดรนในพิกัดโลก (lat, lon, alt) จากภาพและข้อมูลกล้อง
Output: p2_localization.csv

Format: img_file, center_x, center_y, w, h, theta, drone_lat, drone_lon, drone_alt
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import argparse
from problem1_detection import DroneDetectorOBB


class DroneLocalizer:
    """คำนวณตำแหน่งโดรนในพิกัดจริง (lat, lon, alt)"""
    
    def __init__(self, camera_fov_h=60.0, camera_fov_v=45.0):
        """
        Args:
            camera_fov_h: Horizontal Field of View (degrees)
            camera_fov_v: Vertical Field of View (degrees)
        """
        self.camera_fov_h = camera_fov_h
        self.camera_fov_v = camera_fov_v
        
    def estimate_drone_position_improved(self, 
                                         center_x: float, 
                                         center_y: float,
                                         bbox_w: float,
                                         bbox_h: float,
                                         camera_lat: float,
                                         camera_lon: float,
                                         camera_alt: float,
                                         camera_pitch: float = -30.0,
                                         camera_yaw: float = 0.0,
                                         estimated_drone_alt: float = None) -> Tuple[float, float, float]:
        """
        ประมาณตำแหน่งโดรนด้วยวิธีที่แม่นยำขึ้น
        
        Improvements:
        - ใช้ arctan สำหรับคำนวณมุมจาก FOV
        - ประมาณความสูงโดรนจากขนาด bbox (ถ้าไม่ระบุ)
        - เพิ่ม error handling
        - ใช้ WGS84 ellipsoid แทน sphere approximation
        
        Args:
            center_x, center_y: จุดกึ่งกลางโดรนในภาพ (normalized 0-1)
            bbox_w, bbox_h: ขนาด bounding box (ใช้ประมาณระยะทาง)
            camera_lat, camera_lon, camera_alt: ตำแหน่งกล้อง
            camera_pitch: มุมเงยของกล้อง (degrees, + = เงยขึ้น, - = ก้มลง)
            camera_yaw: ทิศทางกล้อง (degrees, 0 = North, 90 = East)
            estimated_drone_alt: ความสูงโดรน (ถ้าเป็น None จะประมาณจาก bbox)
            
        Returns:
            (drone_lat, drone_lon, drone_alt)
        """
        # 1. คำนวณมุมเบี่ยงเบนจากจุดกึ่งกลางภาพ (ใช้ arctan)
        # FOV → Focal length equivalent
        focal_length_h = 0.5 / np.tan(np.radians(self.camera_fov_h / 2))
        focal_length_v = 0.5 / np.tan(np.radians(self.camera_fov_v / 2))
        
        # Pixel offset จากจุดกึ่งกลาง
        offset_x = center_x - 0.5  # -0.5 ถึง +0.5
        offset_y = center_y - 0.5
        
        # คำนวณมุมด้วย arctan
        angle_h = np.degrees(np.arctan(offset_x / focal_length_h))
        angle_v = np.degrees(np.arctan(-offset_y / focal_length_v))  # - เพราะ y เพิ่มลงล่าง
        
        # 2. คำนวณมุมจริง
        actual_pitch = camera_pitch + angle_v
        actual_yaw = camera_yaw + angle_h
        
        # Validate pitch
        if actual_pitch >= 0:
            print(f"Warning: pitch={actual_pitch:.1f}° (looking up), drone position may be inaccurate")
        if abs(actual_pitch) > 85:
            print(f"Warning: extreme pitch angle={actual_pitch:.1f}°, using fallback calculation")
            actual_pitch = max(min(actual_pitch, -5), -85)  # clamp
        
        # 3. ประมาณความสูงโดรน (ถ้าไม่ระบุ)
        if estimated_drone_alt is None:
            # ใช้ขนาด bbox ประมาณ - โดรนใหญ่ = ใกล้กว่า = ต่ำกว่า
            # สูตรประมาณ: altitude ∝ 1/bbox_size
            avg_bbox_size = (bbox_w + bbox_h) / 2
            if avg_bbox_size > 0.1:  # ใหญ่มาก = ต่ำ
                estimated_drone_alt = camera_alt + 30
            elif avg_bbox_size > 0.05:  # ปานกลาง
                estimated_drone_alt = camera_alt + 60
            else:  # เล็ก = สูง
                estimated_drone_alt = camera_alt + 100
            
            print(f"  Auto-estimated altitude: {estimated_drone_alt:.1f}m (based on bbox size {avg_bbox_size:.4f})")
        
        # 4. คำนวณระยะทางแนวนอน
        height_diff = estimated_drone_alt - camera_alt
        
        if height_diff <= 0:
            print(f"Warning: drone altitude ({estimated_drone_alt}m) <= camera altitude ({camera_alt}m)")
            height_diff = 10.0  # fallback
        
        tan_pitch = np.tan(np.radians(actual_pitch))
        if abs(tan_pitch) < 0.001:
            horizontal_dist = 1000.0
            print(f"Warning: near-zero pitch, using default distance {horizontal_dist}m")
        else:
            horizontal_dist = abs(height_diff / tan_pitch)
        
        # Sanity check
        if horizontal_dist > 10000:  # > 10km
            print(f"Warning: unrealistic distance {horizontal_dist:.0f}m, clamping to 2000m")
            horizontal_dist = 2000
        
        # 5. แปลงเป็น GPS coordinates (WGS84 ellipsoid)
        # Reference: https://en.wikipedia.org/wiki/Geographic_coordinate_system
        
        # Earth radius at latitude
        lat_rad = np.radians(camera_lat)
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis (meters)
        f = 1/298.257223563  # flattening
        e2 = 2*f - f*f  # eccentricity squared
        
        # Radius of curvature
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # Meters per degree
        meters_per_deg_lat = (np.pi * N * (1 - e2)) / 180
        meters_per_deg_lon = (np.pi * N * np.cos(lat_rad)) / 180
        
        # 6. คำนวณการเปลี่ยนแปลง
        delta_north = horizontal_dist * np.cos(np.radians(actual_yaw))
        delta_east = horizontal_dist * np.sin(np.radians(actual_yaw))
        
        drone_lat = camera_lat + (delta_north / meters_per_deg_lat)
        drone_lon = camera_lon + (delta_east / meters_per_deg_lon)
        drone_alt = estimated_drone_alt
        
        return drone_lat, drone_lon, drone_alt
    
    def estimate_with_triangulation(self,
                                    center_x: float,
                                    center_y: float,
                                    camera_lat: float,
                                    camera_lon: float,
                                    camera_alt: float,
                                    camera_matrix: np.ndarray = None,
                                    dist_coeffs: np.ndarray = None) -> Tuple[float, float, float]:
        """
        วิธีที่ซับซ้อนกว่า: ใช้ camera calibration และ triangulation
        
        Note: ต้องมี camera intrinsics และอาจต้องมีข้อมูลเพิ่มเติม
        """
        # TODO: Implement advanced triangulation method
        # ต้องการ: camera intrinsics, extrinsics, หรือข้อมูลจาก multiple views
        raise NotImplementedError("Advanced triangulation not yet implemented")
    
    def process_dataset(self,
                       detection_csv: str,
                       metadata_csv: str,
                       output_csv: str = 'p2_localization.csv',
                       camera_pitch: float = -30.0,
                       camera_yaw: float = 0.0,
                       default_drone_alt: float = None,
                       auto_estimate_altitude: bool = True) -> pd.DataFrame:
        """
        ประมวลผลทั้ง dataset
        
        Args:
            detection_csv: ไฟล์ผลลัพธ์จาก problem 1 (p1_detection_obb.csv)
            metadata_csv: ไฟล์ข้อมูลกล้อง (img_file, img_lat, img_lon, img_alt)
            output_csv: ไฟล์ผลลัพธ์
            camera_pitch: มุมเงยกล้อง (ถ้าไม่มีใน metadata)
            camera_yaw: ทิศทางกล้อง (ถ้าไม่มีใน metadata)
            default_drone_alt: altitude โดรนเริ่มต้น (None = auto-estimate)
            auto_estimate_altitude: ประมาณความสูงจาก bbox size
            
        Returns:
            pandas DataFrame
        """
        # อ่านข้อมูล
        df_det = pd.read_csv(detection_csv)
        df_meta = pd.read_csv(metadata_csv)
        
        print(f"Loaded {len(df_det)} detections from {detection_csv}")
        print(f"Loaded {len(df_meta)} metadata entries from {metadata_csv}")
        
        # Merge detection กับ metadata
        df = df_det.merge(df_meta, on='img_file', how='left')
        
        # ตรวจสอบว่ามี metadata ครบหรือไม่
        missing = df[df['img_lat'].isna()]
        if len(missing) > 0:
            print(f"Warning: {len(missing)} detections missing metadata")
        
        # คำนวณตำแหน่งโดรน
        results = []
        
        print(f"\nProcessing {len(df)} detections...")
        print(f"Camera parameters: pitch={camera_pitch}°, yaw={camera_yaw}°")
        
        for idx, row in df.iterrows():
            if pd.notna(row['img_lat']):
                # ใช้ pitch/yaw จาก metadata ถ้ามี
                pitch = row.get('camera_pitch', camera_pitch)
                yaw = row.get('camera_yaw', camera_yaw)
                
                # ประมาณ altitude โดรน
                if auto_estimate_altitude:
                    est_alt = None  # ให้ function ประมาณเอง
                else:
                    est_alt = row.get('drone_alt_estimate', default_drone_alt)
                
                print(f"\nDrone {idx+1}/{len(df)}: {row['img_file']} at ({row['center_x']:.3f}, {row['center_y']:.3f})")
                
                # คำนวณตำแหน่ง (ใช้ฟังก์ชันใหม่)
                drone_lat, drone_lon, drone_alt = self.estimate_drone_position_improved(
                    center_x=row['center_x'],
                    center_y=row['center_y'],
                    bbox_w=row['w'],
                    bbox_h=row['h'],
                    camera_lat=row['img_lat'],
                    camera_lon=row['img_lon'],
                    camera_alt=row['img_alt'],
                    camera_pitch=pitch,
                    camera_yaw=yaw,
                    estimated_drone_alt=est_alt
                )
                
                results.append({
                    'img_file': row['img_file'],
                    'center_x': row['center_x'],
                    'center_y': row['center_y'],
                    'w': row['w'],
                    'h': row['h'],
                    'theta': row['theta'],
                    'drone_lat': round(drone_lat, 6),
                    'drone_lon': round(drone_lon, 6),
                    'drone_alt': round(drone_alt, 2)
                })
        
        # สร้าง DataFrame
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Localization complete!")
        print(f"  - Processed: {len(df_out)} drones")
        print(f"  - Output saved to: {output_csv}")
        print(f"{'='*60}\n")
        
        return df_out


def main():
    parser = argparse.ArgumentParser(description='Problem 2: Drone Localization')
    parser.add_argument('--detection', type=str, default='p1_detection_obb.csv',
                       help='Detection CSV from Problem 1')
    parser.add_argument('--metadata', type=str, default='image_meta.csv',
                       help='Image metadata CSV (img_file, img_lat, img_lon, img_alt)')
    parser.add_argument('--output', type=str, default='p2_localization.csv',
                       help='Output CSV file')
    parser.add_argument('--pitch', type=float, default=-30.0,
                       help='Camera pitch angle (degrees, negative=looking down)')
    parser.add_argument('--yaw', type=float, default=0.0,
                       help='Camera yaw angle (degrees, 0=North)')
    parser.add_argument('--drone-alt', type=float, default=None,
                       help='Fixed drone altitude (meters). If not set, auto-estimate from bbox size')
    parser.add_argument('--no-auto-alt', action='store_true',
                       help='Disable automatic altitude estimation')
    parser.add_argument('--fov-h', type=float, default=60.0,
                       help='Camera horizontal FOV (degrees)')
    parser.add_argument('--fov-v', type=float, default=45.0,
                       help='Camera vertical FOV (degrees)')
    
    args = parser.parse_args()
    
    # สร้าง localizer
    localizer = DroneLocalizer(camera_fov_h=args.fov_h, camera_fov_v=args.fov_v)
    
    # ประมวลผล
    df = localizer.process_dataset(
        detection_csv=args.detection,
        metadata_csv=args.metadata,
        output_csv=args.output,
        camera_pitch=args.pitch,
        camera_yaw=args.yaw,
        default_drone_alt=args.drone_alt,
        auto_estimate_altitude=(not args.no_auto_alt)
    )
    
    # แสดงผลลัพธ์
    if len(df) > 0:
        print("\n" + "="*60)
        print("Sample results (first 5 rows):")
        print(df.head(5).to_string(index=False))
        print("="*60)


if __name__ == '__main__':
    main()
