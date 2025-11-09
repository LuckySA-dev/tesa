"""
Validation Script for Problem 2
================================
ตรวจสอบความถูกต้องและ sanity check
"""

import pandas as pd
import numpy as np

def validate_localization_results(csv_file: str):
    """ตรวจสอบผลลัพธ์ localization"""
    
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*70}")
    print(f"VALIDATION REPORT: {csv_file}")
    print(f"{'='*70}\n")
    
    # 1. ตรวจสอบ format
    required_cols = ['img_file', 'center_x', 'center_y', 'w', 'h', 'theta', 
                     'drone_lat', 'drone_lon', 'drone_alt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ FAIL: Missing columns: {missing_cols}")
        return False
    else:
        print(f"✅ Format: All required columns present")
    
    # 2. ตรวจสอบค่า normalized
    if not ((df['center_x'] >= 0) & (df['center_x'] <= 1)).all():
        print(f"❌ FAIL: center_x out of range [0,1]")
        return False
    if not ((df['center_y'] >= 0) & (df['center_y'] <= 1)).all():
        print(f"❌ FAIL: center_y out of range [0,1]")
        return False
    if not ((df['w'] >= 0) & (df['w'] <= 1)).all():
        print(f"❌ FAIL: w out of range [0,1]")
        return False
    if not ((df['h'] >= 0) & (df['h'] <= 1)).all():
        print(f"❌ FAIL: h out of range [0,1]")
        return False
    
    print(f"✅ Normalized coords: All in valid range [0,1]")
    
    # 3. ตรวจสอบ theta
    if not ((df['theta'] >= -90) & (df['theta'] <= 90)).all():
        print(f"❌ FAIL: theta out of range [-90,90]")
        return False
    
    print(f"✅ Theta: All in valid range [-90,90]°")
    
    # 4. ตรวจสอบ GPS coordinates (reasonable range for Thailand)
    if not ((df['drone_lat'] >= 5) & (df['drone_lat'] <= 21)).all():
        print(f"⚠️  WARNING: latitude outside Thailand (5-21°N)")
    else:
        print(f"✅ Latitude: Reasonable range")
    
    if not ((df['drone_lon'] >= 97) & (df['drone_lon'] <= 106)).all():
        print(f"⚠️  WARNING: longitude outside Thailand (97-106°E)")
    else:
        print(f"✅ Longitude: Reasonable range")
    
    # 5. ตรวจสอบ altitude
    if (df['drone_alt'] < 0).any():
        print(f"❌ FAIL: Negative altitude detected")
        return False
    if (df['drone_alt'] > 500).any():
        print(f"⚠️  WARNING: Very high altitude (>500m) - drone or airplane?")
    else:
        print(f"✅ Altitude: Reasonable range")
    
    # 6. ตรวจสอบระยะห่างระหว่างโดรน
    print(f"\n📊 STATISTICS:")
    print(f"  Total drones: {len(df)}")
    print(f"  Altitude range: {df['drone_alt'].min():.1f}m - {df['drone_alt'].max():.1f}m")
    print(f"  Altitude mean: {df['drone_alt'].mean():.1f}m ± {df['drone_alt'].std():.1f}m")
    
    # คำนวณระยะทางระหว่างโดรน
    if len(df) > 1:
        print(f"\n📏 DISTANCE BETWEEN DRONES:")
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                # Haversine distance
                lat1, lon1 = np.radians(df.iloc[i]['drone_lat']), np.radians(df.iloc[i]['drone_lon'])
                lat2, lon2 = np.radians(df.iloc[j]['drone_lat']), np.radians(df.iloc[j]['drone_lon'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                horizontal_dist = 6371000 * c  # meters
                
                vertical_dist = abs(df.iloc[i]['drone_alt'] - df.iloc[j]['drone_alt'])
                total_dist = np.sqrt(horizontal_dist**2 + vertical_dist**2)
                
                print(f"  Drone {i+1} ↔ Drone {j+1}: {horizontal_dist:.1f}m horizontal, "
                      f"{vertical_dist:.1f}m vertical, {total_dist:.1f}m total")
    
    # 7. ตรวจสอบความสมเหตุสมผล
    print(f"\n🔍 SANITY CHECKS:")
    
    # โดรนควรไม่ใกล้กันเกินไป (ป้องกันชนกัน)
    min_safe_distance = 5.0  # meters
    if len(df) > 1:
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                lat1, lon1 = np.radians(df.iloc[i]['drone_lat']), np.radians(df.iloc[i]['drone_lon'])
                lat2, lon2 = np.radians(df.iloc[j]['drone_lat']), np.radians(df.iloc[j]['drone_lon'])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                dist = 6371000 * c
                
                if dist < min_safe_distance:
                    print(f"  ⚠️  Drone {i+1} and {j+1} very close ({dist:.1f}m) - possible collision risk")
    
    print(f"\n{'='*70}")
    print(f"✅ VALIDATION PASSED")
    print(f"{'='*70}\n")
    
    return True


if __name__ == '__main__':
    import sys
    
    files = ['p2_localization.csv', 'p2_localization_v2.csv', 'p2_localization_final.csv']
    
    for csv_file in files:
        try:
            validate_localization_results(csv_file)
        except FileNotFoundError:
            print(f"⚠️  File not found: {csv_file}\n")
        except Exception as e:
            print(f"❌ Error validating {csv_file}: {e}\n")
