"""
Centroid Tracker for Object Tracking
=====================================
Track objects across frames using centroid distance matching.
Based on: https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

Features:
- Assign unique ID to each object
- Track objects across frames using Euclidean distance
- Handle disappeared objects
- Store tracking path history
- Calculate velocity and direction

Author: TESA Defence Team
Date: November 8, 2025
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque


class CentroidTracker:
    """Track objects using centroid distance matching"""
    
    def __init__(self, max_disappeared=50, max_distance=100, track_history=30):
        """
        Initialize Centroid Tracker
        
        Args:
            max_disappeared (int): Max frames object can disappear before deregistering
            max_distance (float): Max pixel distance to match same object
            track_history (int): Number of frames to keep in path history
        """
        self.nextObjectID = 1  # Start from 1 instead of 0 (as per requirements)
        self.objects = OrderedDict()  # {id: (cx, cy)}
        self.disappeared = OrderedDict()  # {id: frame_count}
        self.track_paths = OrderedDict()  # {id: deque([(x,y), ...])}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_history = track_history
        
        # For velocity calculation
        self.previous_centroids = OrderedDict()  # {id: (cx, cy, timestamp)}
        
    def register(self, centroid, timestamp=None):
        """
        Register new object with unique ID
        
        Args:
            centroid (tuple): (center_x, center_y)
            timestamp (float): Current timestamp for velocity tracking
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.track_paths[self.nextObjectID] = deque(maxlen=self.track_history)
        self.track_paths[self.nextObjectID].append(centroid)
        
        if timestamp is not None:
            self.previous_centroids[self.nextObjectID] = (*centroid, timestamp)
        
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        """
        Remove object from tracking
        
        Args:
            objectID (int): Object ID to remove
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.track_paths:
            del self.track_paths[objectID]
        if objectID in self.previous_centroids:
            del self.previous_centroids[objectID]
    
    def update(self, detections, timestamp=None):
        """
        Update tracker with new detections
        
        Args:
            detections (list): List of (center_x, center_y) tuples in pixels
            timestamp (float): Current timestamp for velocity calculation
            
        Returns:
            OrderedDict: {objectID: (cx, cy)}
        """
        # No detections - mark all as disappeared
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # Deregister if disappeared too long
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            
            return self.objects
        
        # Initialize centroids array
        input_centroids = np.array(detections)
        
        # No existing objects - register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(tuple(centroid), timestamp)
        
        else:
            # Get existing object IDs and centroids
            objectIDs = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            # Compute pairwise Euclidean distances
            D = dist.cdist(object_centroids, input_centroids)
            
            # Find minimum distances
            # rows = object indices sorted by min distance
            # cols = detection indices with min distance for each object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Track which rows/cols are used
            used_rows = set()
            used_cols = set()
            
            # Match existing objects to new detections
            for (row, col) in zip(rows, cols):
                # Skip if already used
                if row in used_rows or col in used_cols:
                    continue
                
                # Skip if distance too large (likely different object)
                if D[row, col] > self.max_distance:
                    continue
                
                # Update object
                objectID = objectIDs[row]
                new_centroid = tuple(input_centroids[col])
                
                self.objects[objectID] = new_centroid
                self.disappeared[objectID] = 0
                
                # Update path
                self.track_paths[objectID].append(new_centroid)
                
                # Update velocity tracking
                if timestamp is not None:
                    self.previous_centroids[objectID] = (*new_centroid, timestamp)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects (no match found)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            
            # Register new objects (new detections)
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), timestamp)
        
        return self.objects
    
    def get_track_path(self, objectID):
        """
        Get tracking path for specific object
        
        Args:
            objectID (int): Object ID
            
        Returns:
            list: List of (x, y) tuples representing path
        """
        if objectID in self.track_paths:
            return list(self.track_paths[objectID])
        return []
    
    def get_all_paths(self):
        """
        Get all tracking paths
        
        Returns:
            dict: {objectID: [(x, y), ...]}
        """
        return {oid: list(path) for oid, path in self.track_paths.items()}
    
    def calculate_velocity(self, objectID, current_time, fps=30, pixels_per_meter=100):
        """
        Calculate velocity and direction of object
        
        Args:
            objectID (int): Object ID
            current_time (float): Current timestamp
            fps (int): Frames per second (for time calculation)
            pixels_per_meter (float): Conversion factor from pixels to meters
            
        Returns:
            dict: {
                'speed': float (m/s),
                'direction': float (degrees, 0=right, 90=up, 180=left, 270=down),
                'distance_pixels': float
            }
        """
        if objectID not in self.previous_centroids or objectID not in self.objects:
            return {'speed': 0.0, 'direction': 0.0, 'distance_pixels': 0.0}
        
        # Get previous and current positions
        prev_x, prev_y, prev_time = self.previous_centroids[objectID]
        curr_x, curr_y = self.objects[objectID]
        
        # Calculate Euclidean distance
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Calculate time difference
        time_diff = current_time - prev_time if current_time != prev_time else 1.0/fps
        
        # Calculate speed (pixels/second -> m/s)
        speed_pixels_per_sec = distance_pixels / time_diff if time_diff > 0 else 0
        speed_m_per_sec = speed_pixels_per_sec / pixels_per_meter
        
        # Calculate direction (degrees, 0=right, 90=up, 180=left, 270=down)
        # Negative dy because y increases downward in image coordinates
        direction = np.degrees(np.arctan2(-dy, dx))
        if direction < 0:
            direction += 360
        
        return {
            'speed': round(speed_m_per_sec, 2),
            'direction': round(direction, 2),
            'distance_pixels': round(distance_pixels, 2)
        }
    
    def get_total_tracked_objects(self):
        """Get total number of objects ever tracked"""
        return self.nextObjectID
    
    def get_active_objects(self):
        """Get number of currently active objects"""
        return len(self.objects)
    
    def reset(self):
        """Reset tracker to initial state"""
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.track_paths = OrderedDict()
        self.previous_centroids = OrderedDict()


if __name__ == '__main__':
    # Test the tracker
    print("Testing CentroidTracker...")
    
    tracker = CentroidTracker(max_disappeared=10, max_distance=50)
    
    # Simulate detections over frames
    test_detections = [
        [(100, 100), (200, 200)],  # Frame 1: 2 objects
        [(105, 105), (205, 205)],  # Frame 2: same objects moved
        [(110, 110)],               # Frame 3: 1 object disappeared
        [(115, 115), (300, 300)],  # Frame 4: 1 returned, 1 new
    ]
    
    for frame_num, detections in enumerate(test_detections, 1):
        print(f"\nFrame {frame_num}: {len(detections)} detections")
        objects = tracker.update(detections, timestamp=frame_num)
        
        for obj_id, centroid in objects.items():
            print(f"  Object {obj_id}: {centroid}")
    
    print(f"\nTotal tracked objects: {tracker.get_total_tracked_objects()}")
    print(f"Currently active: {tracker.get_active_objects()}")
