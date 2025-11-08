from problem1_detection import DroneDetectorOBB

detector = DroneDetectorOBB()
detections = detector.detect_single_image('images/drones.jpg')

print(f'After duplicate removal: {len(detections)} detections')
for i, det in enumerate(detections):
    print(f'Drone {i}: x={det["center_x"]:.4f}, y={det["center_y"]:.4f}, theta={det["theta"]:.2f}°')
