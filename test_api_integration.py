"""Test API integration and check payloads"""
from problem1_video_tracking import DroneVideoTracker
import cv2

print("Testing API Integration...")
print("="*60)

# Create tracker with API enabled
tracker = DroneVideoTracker(enable_api=True)

# Process video
tracker.process_video('videos/video_01.mp4', display=False, conf_threshold=0.4)

print("\n" + "="*60)
print("📡 API MOCK DATA ANALYSIS")
print("="*60)

# Check sent data
print(f"\n1. Total API calls: {len(tracker.api.sent_data)}")

# Analyze alarm data
alarm_data = [d for d in tracker.api.sent_data if d['endpoint'] == '/alarm']
if alarm_data:
    print(f"\n2. First Alarm Data:")
    alarm = alarm_data[0]
    print(f"   ✅ Endpoint: {alarm['endpoint']}")
    print(f"   ✅ Drone count: {alarm['payload']['drone_count']}")
    print(f"   ✅ Has image: {'image_base64' in alarm['payload']}")
    if 'image_base64' in alarm['payload']:
        img_size = len(alarm['payload']['image_base64'])
        print(f"   ✅ Image size: {img_size:,} bytes ({img_size/1024:.1f} KB)")

# Analyze tracking data
tracking_data = [d for d in tracker.api.sent_data if d['endpoint'] == '/tracking']
if tracking_data:
    print(f"\n3. Tracking Data:")
    print(f"   ✅ Total tracking calls: {len(tracking_data)}")
    print(f"   ✅ First call objects: {len(tracking_data[0]['payload']['object'])}")
    print(f"   ✅ Last call objects: {len(tracking_data[-1]['payload']['object'])}")
    print(f"   ✅ Has image: {'image_base64' in tracking_data[0]['payload']}")
    if 'image_base64' in tracking_data[0]['payload']:
        img_size = len(tracking_data[0]['payload']['image_base64'])
        print(f"   ✅ Image size: {img_size:,} bytes ({img_size/1024:.1f} KB)")

print("\n" + "="*60)
print("✅ API INTEGRATION TEST COMPLETE")
print("="*60)
