"""
Analyze Problem 3 Tracking Results
===================================
ตรวจสอบคุณภาพของการ tracking และหาจุดที่ต้องปรับปรุง
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_tracking(csv_file: str):
    """วิเคราะห์ผลลัพธ์ tracking"""
    
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*70}")
    print(f"TRACKING QUALITY ANALYSIS: {csv_file}")
    print(f"{'='*70}\n")
    
    # 1. ข้อมูลพื้นฐาน
    total_frames = df['frame_id'].max() + 1
    unique_tracks = df['track_id'].nunique()
    total_detections = len(df)
    
    print(f"📊 BASIC STATISTICS:")
    print(f"  Total frames: {total_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Unique track IDs: {unique_tracks}")
    print(f"  Average detections per frame: {total_detections / total_frames:.2f}")
    
    # 2. วิเคราะห์แต่ละ track
    print(f"\n📈 TRACK DETAILS:")
    track_stats = []
    
    for tid in sorted(df['track_id'].unique()):
        track_df = df[df['track_id'] == tid]
        frames = track_df['frame_id'].values
        
        first_frame = frames.min()
        last_frame = frames.max()
        duration = last_frame - first_frame + 1
        appearances = len(frames)
        gaps = duration - appearances
        
        track_stats.append({
            'track_id': tid,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'duration': duration,
            'appearances': appearances,
            'gaps': gaps,
            'gap_rate': gaps / duration if duration > 0 else 0
        })
        
        status = "✅" if gaps == 0 else f"⚠️  ({gaps} gaps)"
        print(f"  Track {tid:2d}: frames {first_frame:3d}-{last_frame:3d} "
              f"({appearances:3d}/{duration:3d} frames) {status}")
    
    track_df = pd.DataFrame(track_stats)
    
    # 3. ตรวจสอบปัญหา
    print(f"\n🔍 ISSUE DETECTION:")
    
    issues = []
    
    # 3.1 Short tracks (อาจเป็น false positive หรือ ID switch)
    short_tracks = track_df[track_df['appearances'] < 10]
    if len(short_tracks) > 0:
        issues.append(f"SHORT TRACKS: {len(short_tracks)} tracks with <10 frames")
        print(f"  ⚠️  {len(short_tracks)} short tracks (<10 frames): {short_tracks['track_id'].tolist()}")
    else:
        print(f"  ✅ No short tracks detected")
    
    # 3.2 Tracks with many gaps (tracking loss)
    gappy_tracks = track_df[track_df['gap_rate'] > 0.3]
    if len(gappy_tracks) > 0:
        issues.append(f"GAPPY TRACKS: {len(gappy_tracks)} tracks with >30% gaps")
        print(f"  ⚠️  {len(gappy_tracks)} tracks with >30% gaps: {gappy_tracks['track_id'].tolist()}")
    else:
        print(f"  ✅ No gappy tracks detected")
    
    # 3.3 ID switches (tracks ที่ซ้อนทับกัน)
    print(f"\n🔄 ID SWITCH DETECTION:")
    potential_switches = 0
    
    for i, row1 in track_df.iterrows():
        for j, row2 in track_df.iterrows():
            if i >= j:
                continue
            
            # ถ้า track ทับซ้อนกัน = อาจเป็น ID switch
            if not (row1['last_frame'] < row2['first_frame'] or 
                   row2['last_frame'] < row1['first_frame']):
                # Check if they appear in same frames
                frames1 = set(df[df['track_id'] == row1['track_id']]['frame_id'])
                frames2 = set(df[df['track_id'] == row2['track_id']]['frame_id'])
                overlap = len(frames1 & frames2)
                
                if overlap == 0:
                    # ไม่อยู่เฟรมเดียวกัน แต่เวลาทับ = อาจเป็น ID switch
                    potential_switches += 1
                    print(f"  ⚠️  Potential switch: Track {row1['track_id']} → Track {row2['track_id']}")
    
    if potential_switches == 0:
        print(f"  ✅ No obvious ID switches detected")
    else:
        issues.append(f"ID SWITCHES: {potential_switches} potential switches")
    
    # 3.4 Expected vs Actual tracks
    # สมมติว่าในภาพมี 4 โดรน ควรมี ~4 tracks
    expected_tracks = 4  # จากภาพ drones.jpg
    
    if unique_tracks > expected_tracks * 1.5:
        issues.append(f"TOO MANY TRACKS: {unique_tracks} tracks (expected ~{expected_tracks})")
        print(f"\n  ⚠️  Too many tracks: {unique_tracks} (expected ~{expected_tracks})")
        print(f"     → Possible causes: ID switches, false positives, duplicate detections")
    elif unique_tracks < expected_tracks * 0.5:
        issues.append(f"TOO FEW TRACKS: {unique_tracks} tracks (expected ~{expected_tracks})")
        print(f"\n  ⚠️  Too few tracks: {unique_tracks} (expected ~{expected_tracks})")
        print(f"     → Possible causes: missed detections, tracks merged incorrectly")
    else:
        print(f"\n  ✅ Track count reasonable: {unique_tracks} (expected ~{expected_tracks})")
    
    # 4. Detections per frame
    print(f"\n📊 DETECTIONS PER FRAME:")
    dets_per_frame = df.groupby('frame_id').size()
    print(f"  Min: {dets_per_frame.min()}")
    print(f"  Max: {dets_per_frame.max()}")
    print(f"  Mean: {dets_per_frame.mean():.2f}")
    print(f"  Std: {dets_per_frame.std():.2f}")
    
    if dets_per_frame.max() > expected_tracks * 1.5:
        issues.append(f"TOO MANY DETECTIONS: max {dets_per_frame.max()} per frame")
        print(f"  ⚠️  Some frames have too many detections (max: {dets_per_frame.max()})")
    
    # 5. สรุปและคำแนะนำ
    print(f"\n{'='*70}")
    if len(issues) == 0:
        print(f"✅ TRACKING QUALITY: GOOD")
    else:
        print(f"⚠️  TRACKING QUALITY: NEEDS IMPROVEMENT")
        print(f"\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if len(short_tracks) > 0:
        print(f"  1. Increase --track-thresh to 0.6-0.7 (filter low-confidence detections)")
    
    if len(gappy_tracks) > 0:
        print(f"  2. Decrease --conf to 0.2 (detect more consistently)")
        print(f"  3. Increase track_buffer in ByteTracker (allow longer gaps)")
    
    if potential_switches > 0:
        print(f"  4. Decrease --match-thresh to 0.3-0.4 (more tolerant matching)")
        print(f"  5. Add appearance features (ReID model) for better matching")
    
    if unique_tracks > expected_tracks * 1.5:
        print(f"  6. Enable duplicate removal post-processing")
        print(f"  7. Tune NMS parameters (--iou in YOLO)")
    
    print(f"\n{'='*70}\n")
    
    return track_df, issues


if __name__ == '__main__':
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'p3_tracking_obb.csv'
    
    try:
        track_df, issues = analyze_tracking(csv_file)
        
        # บันทึก track statistics
        output_file = csv_file.replace('.csv', '_analysis.csv')
        track_df.to_csv(output_file, index=False)
        print(f"Track statistics saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
    except Exception as e:
        print(f"Error: {e}")
