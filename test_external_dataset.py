"""
Test External Dataset (Kaggle Drone Dataset)
Download and test system with external drone detection datasets
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import shutil


class ExternalDatasetTester:
    """Test system with external datasets"""
    
    def __init__(self, data_dir='external_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_sample_videos(self):
        """
        Download sample drone videos for testing
        
        Note: In production, use Kaggle API:
        kaggle datasets download -d <dataset-name>
        """
        print("="*70)
        print("📦 EXTERNAL DATASET DOWNLOADER")
        print("="*70)
        
        # Sample URLs (replace with actual Kaggle dataset)
        # For now, we'll create test videos with different resolutions
        
        print("\n⚠️  Note: For real Kaggle datasets, use:")
        print("   pip install kaggle")
        print("   kaggle datasets download -d dataset-name")
        print()
        
        # Create test videos with different resolutions
        test_configs = [
            {'name': '720p_video.mp4', 'width': 1280, 'height': 720, 'fps': 30},
            {'name': '1080p_video.mp4', 'width': 1920, 'height': 1080, 'fps': 30},
            {'name': '4k_video.mp4', 'width': 3840, 'height': 2160, 'fps': 60},
        ]
        
        print("📹 Creating test videos with different resolutions...")
        for config in test_configs:
            self._create_test_video(config)
        
        print("\n✅ Test videos created successfully!")
        
    def _create_test_video(self, config):
        """Create a test video with specified resolution"""
        output_path = self.data_dir / config['name']
        
        if output_path.exists():
            print(f"   ⏭️  Skipping {config['name']} (already exists)")
            return
        
        print(f"   🎬 Creating {config['name']} ({config['width']}x{config['height']} @ {config['fps']} FPS)...")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            config['fps'], 
            (config['width'], config['height'])
        )
        
        # Generate 30 frames with moving objects (simulating drones)
        num_frames = 30
        for i in range(num_frames):
            # Create blank frame
            frame = np.random.randint(100, 150, (config['height'], config['width'], 3), dtype=np.uint8)
            
            # Add 2-3 "drones" (colored rectangles) moving across frame
            num_objects = np.random.randint(2, 4)
            for j in range(num_objects):
                # Calculate position (moving left to right)
                x = int((i / num_frames) * config['width'] + j * 200)
                y = int(config['height'] * (0.3 + j * 0.2))
                w, h = 60, 40
                
                # Draw rectangle (simulating drone)
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
                
                # Add some noise
                cv2.circle(frame, (x+w//2, y+h//2), 5, (255, 255, 255), -1)
            
            writer.write(frame)
        
        writer.release()
        
    def test_video(self, video_path, output_prefix='external'):
        """Test complete pipeline with external video"""
        video_path = Path(video_path)
        
        print("\n" + "="*70)
        print(f"🧪 TESTING: {video_path.name}")
        print("="*70)
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"\n📹 Video Properties:")
        print(f"   • Resolution: {width}x{height}")
        print(f"   • FPS: {fps}")
        print(f"   • Frames: {total_frames}")
        print(f"   • Duration: {total_frames/fps:.1f}s")
        
        # Create output filenames
        p1_output = f"submissions/{output_prefix}_p1.csv"
        p2_temp_output = f"submissions/{output_prefix}_p2_temp.csv"
        p2_output = f"submissions/{output_prefix}_p2.csv"
        p3_output = f"submissions/{output_prefix}_p3.csv"
        
        # Step 1: Auto-tune confidence (optional but recommended)
        print(f"\n🎯 Step 1: Auto-tuning confidence threshold...")
        try:
            from auto_tune_confidence import find_optimal_confidence
            optimal_conf = find_optimal_confidence(
                video_path=str(video_path),
                num_sample_frames=min(10, total_frames),
                target_detections_per_frame=2.0
            )
            print(f"   ✅ Optimal confidence: {optimal_conf}")
        except Exception as e:
            print(f"   ⚠️  Auto-tune failed: {e}")
            print(f"   ℹ️  Using default confidence: 0.5")
            optimal_conf = 0.5
        
        # Step 2: Problem 1 - Detection
        print(f"\n📊 Step 2: Running Problem 1 (Detection + Tracking)...")
        cmd_p1 = f'python problem1_competition.py --video "{video_path}" --output "{p1_output}" --conf {optimal_conf}'
        result = os.system(cmd_p1)
        
        if result != 0:
            print(f"   ❌ Problem 1 failed!")
            return False
        
        # Check output
        if not Path(p1_output).exists():
            print(f"   ❌ Output file not created: {p1_output}")
            return False
        
        df_p1 = pd.read_csv(p1_output)
        print(f"   ✅ Problem 1 complete: {len(df_p1)} detections")
        
        # Step 3: Problem 2 - Inference
        print(f"\n🔮 Step 3: Running Problem 2 (Inference)...")
        cmd_p2 = f'python problem2_inference.py --detections "{p1_output}" --output "{p2_temp_output}" --video "{video_path}"'
        result = os.system(cmd_p2)
        
        if result != 0:
            print(f"   ❌ Problem 2 inference failed!")
            return False
        
        # Step 4: Format conversion
        print(f"\n🔧 Step 4: Converting to Problem 2 format...")
        cmd_p2_fmt = f'python fix_problem2_format.py --input "{p2_temp_output}" --output "{p2_output}"'
        result = os.system(cmd_p2_fmt)
        
        if result != 0:
            print(f"   ❌ Format conversion failed!")
            return False
        
        df_p2 = pd.read_csv(p2_output)
        print(f"   ✅ Problem 2 complete: {len(df_p2)} predictions")
        
        # Step 5: Problem 3 - Integration
        print(f"\n🚀 Step 5: Running Problem 3 (Integration)...")
        cmd_p3 = f'python problem3_integration.py --video "{video_path}" --output "{p3_output}" --conf {optimal_conf}'
        result = os.system(cmd_p3)
        
        if result != 0:
            print(f"   ❌ Problem 3 failed!")
            return False
        
        df_p3 = pd.read_csv(p3_output)
        print(f"   ✅ Problem 3 complete: {len(df_p3)} predictions")
        
        # Summary
        print("\n" + "="*70)
        print("✅ TESTING COMPLETE")
        print("="*70)
        print(f"📊 Results Summary:")
        print(f"   • Video: {video_path.name}")
        print(f"   • Resolution: {width}x{height} @ {fps} FPS")
        print(f"   • Confidence: {optimal_conf}")
        print(f"   • Problem 1: {len(df_p1)} detections")
        print(f"   • Problem 2: {len(df_p2)} predictions")
        print(f"   • Problem 3: {len(df_p3)} predictions")
        print(f"\n📁 Output Files:")
        print(f"   • {p1_output}")
        print(f"   • {p2_output}")
        print(f"   • {p3_output}")
        print("="*70)
        
        return True
    
    def compare_results(self, baseline_files, test_files):
        """Compare results from different resolutions"""
        print("\n" + "="*70)
        print("📊 RESULTS COMPARISON")
        print("="*70)
        
        comparison = []
        
        for test_file in test_files:
            if not Path(test_file).exists():
                continue
            
            df = pd.read_csv(test_file)
            
            # Get video info from filename
            video_name = Path(test_file).stem.split('_')[0]
            
            comparison.append({
                'video': video_name,
                'file': Path(test_file).name,
                'records': len(df),
                'columns': len(df.columns),
                'format': ', '.join(df.columns[:3]) + '...'
            })
        
        if comparison:
            df_comp = pd.DataFrame(comparison)
            print("\n" + df_comp.to_string(index=False))
        else:
            print("   ⚠️  No test files found for comparison")
        
        print("\n" + "="*70)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with external datasets')
    parser.add_argument('--download', action='store_true',
                       help='Download/create test videos')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all videos in external_data/')
    parser.add_argument('--video', type=str,
                       help='Test specific video file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results from all tests')
    
    args = parser.parse_args()
    
    tester = ExternalDatasetTester()
    
    if args.download:
        tester.download_sample_videos()
    
    if args.test_all:
        print("\n🧪 Testing all videos in external_data/...")
        videos = list(tester.data_dir.glob('*.mp4'))
        
        if not videos:
            print("   ⚠️  No videos found. Run with --download first.")
            return
        
        for video in videos:
            prefix = video.stem
            success = tester.test_video(video, output_prefix=prefix)
            if not success:
                print(f"   ⚠️  Testing failed for {video.name}")
    
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Video not found: {args.video}")
            return
        
        prefix = video_path.stem
        tester.test_video(video_path, output_prefix=prefix)
    
    if args.compare:
        # Find all test output files
        test_files = list(Path('submissions').glob('*_p3.csv'))
        tester.compare_results(None, test_files)
    
    if not any([args.download, args.test_all, args.video, args.compare]):
        parser.print_help()


if __name__ == '__main__':
    main()
