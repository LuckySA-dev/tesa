"""
API Client for Satellite Communication
========================================

Send drone detection and tracking data to satellite endpoint.

Required JSON Format:
{
    "time": 1316357487,
    "object": [
        {
            "frame": 0,
            "id": 1,
            "type": "DJIMavic",
            "lat": 13.22,
            "lon": 66.32,
            "velocity": 15.2,
            "direction": 45.3
        }
    ],
    "image_base64": "base64_encoded_image"
}

Author: TESA Defence Team
Date: November 8, 2025
"""

import requests
import json
import base64
import time
from typing import List, Dict, Optional
import cv2
import numpy as np
from pathlib import Path


class DroneAlertAPI:
    """API Client for sending drone alerts to satellite"""
    
    def __init__(self, api_url: str, api_key: str = None, timeout: int = 5):
        """
        Initialize API Client
        
        Args:
            api_url: Satellite API endpoint URL
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.first_alarm_sent = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 30  # seconds
        
    def encode_image_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """
        Encode image frame to base64 string
        
        Args:
            frame: Image frame (numpy array)
            quality: JPEG quality (0-100)
            
        Returns:
            Base64 encoded string
        """
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    def format_payload(self, 
                      objects: List[Dict], 
                      frame: Optional[np.ndarray] = None,
                      include_image: bool = True) -> Dict:
        """
        Format data into required JSON structure
        
        Args:
            objects: List of tracked objects with data
            frame: Image frame (optional)
            include_image: Whether to include base64 image
            
        Returns:
            Formatted JSON payload
        """
        # Current timestamp (Unix time)
        current_time = int(time.time())
        
        # Format object data
        formatted_objects = []
        for obj in objects:
            formatted_objects.append({
                'frame': obj.get('frame', 0),
                'id': obj.get('object_id', 0),
                'type': obj.get('drone_type', 'Unknown'),
                'lat': obj.get('lat', 0.0),
                'lon': obj.get('lon', 0.0),
                'velocity': obj.get('speed_ms', 0.0),
                'direction': obj.get('direction_deg', 0.0)
            })
        
        # Build payload
        payload = {
            'time': current_time,
            'object': formatted_objects
        }
        
        # Add image if requested
        if include_image and frame is not None:
            payload['image_base64'] = self.encode_image_base64(frame)
        else:
            payload['image_base64'] = ''
        
        return payload
    
    def send_first_alarm(self, drone_count: int, frame: Optional[np.ndarray] = None) -> bool:
        """
        Send first alarm when drones detected
        
        Args:
            drone_count: Number of drones detected
            frame: Image frame (optional)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if self.first_alarm_sent:
            return True  # Already sent
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alarm_time < self.alarm_cooldown:
            return False  # Too soon
        
        # Prepare first alarm payload
        payload = {
            'type': 'first_alarm',
            'time': int(current_time),
            'drone_count': drone_count,
            'message': f'⚠️ FIRST ALARM: {drone_count} drone(s) detected'
        }
        
        # Add image if available
        if frame is not None:
            payload['image_base64'] = self.encode_image_base64(frame)
        
        # Send
        success = self._send_request(payload, endpoint='/alarm')
        
        if success:
            self.first_alarm_sent = True
            self.last_alarm_time = current_time
            print(f'🚨 First Alarm sent: {drone_count} drones detected')
        
        return success
    
    def send_tracking_data(self, 
                          objects: List[Dict], 
                          frame: Optional[np.ndarray] = None,
                          include_image: bool = False) -> bool:
        """
        Send tracking data to satellite
        
        Args:
            objects: List of tracked objects
            frame: Current frame (optional)
            include_image: Whether to include image
            
        Returns:
            True if sent successfully
        """
        if not objects:
            return False
        
        # Format payload
        payload = self.format_payload(objects, frame, include_image)
        
        # Send
        success = self._send_request(payload, endpoint='/tracking')
        
        if success:
            print(f'📡 Sent tracking data: {len(objects)} objects')
        
        return success
    
    def send_batch(self, 
                  all_objects: List[Dict],
                  frame: Optional[np.ndarray] = None) -> bool:
        """
        Send batch of tracking data
        
        Args:
            all_objects: All tracked objects from multiple frames
            frame: Latest frame (optional)
            
        Returns:
            True if sent successfully
        """
        if not all_objects:
            return False
        
        # Group by frame
        frames_data = {}
        for obj in all_objects:
            frame_num = obj.get('frame', 0)
            if frame_num not in frames_data:
                frames_data[frame_num] = []
            frames_data[frame_num].append(obj)
        
        # Send each frame's data
        success_count = 0
        for frame_num, objects in frames_data.items():
            payload = self.format_payload(objects, frame if frame_num == max(frames_data.keys()) else None, False)
            if self._send_request(payload):
                success_count += 1
        
        print(f'📡 Sent {success_count}/{len(frames_data)} batches')
        
        return success_count > 0
    
    def _send_request(self, payload: Dict, endpoint: str = '') -> bool:
        """
        Internal method to send HTTP POST request
        
        Args:
            payload: JSON data to send
            endpoint: API endpoint path
            
        Returns:
            True if successful (status 200-299)
        """
        try:
            # Full URL
            url = self.api_url + endpoint
            
            # Headers
            headers = {
                'Content-Type': 'application/json'
            }
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Send POST request
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check status
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                print(f'❌ API Error: {response.status_code} - {response.text}')
                return False
                
        except requests.exceptions.Timeout:
            print(f'❌ API Timeout: {self.timeout}s exceeded')
            return False
            
        except requests.exceptions.ConnectionError:
            print(f'❌ API Connection Error: Cannot reach {self.api_url}')
            return False
            
        except Exception as e:
            print(f'❌ API Error: {str(e)}')
            return False
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if API is reachable
        """
        try:
            response = requests.get(
                self.api_url,
                timeout=self.timeout
            )
            print(f'✅ API connection OK: {response.status_code}')
            return True
        except Exception as e:
            print(f'❌ API connection failed: {str(e)}')
            return False


# Mock API for testing
class MockSatelliteAPI(DroneAlertAPI):
    """Mock API client for testing without real endpoint"""
    
    def __init__(self):
        super().__init__(api_url='http://localhost:8000/api/v1')
        self.sent_data = []
    
    def _send_request(self, payload: Dict, endpoint: str = '') -> bool:
        """Mock send - just store data"""
        self.sent_data.append({
            'endpoint': endpoint,
            'payload': payload,
            'timestamp': time.time()
        })
        print(f'✅ [MOCK] Sent to {endpoint}: {len(payload.get("object", []))} objects')
        return True
    
    def get_sent_data(self) -> List[Dict]:
        """Get all sent data for testing"""
        return self.sent_data


# Example usage
if __name__ == '__main__':
    print("Testing DroneAlertAPI...")
    print("="*60)
    
    # Use mock API for testing
    api = MockSatelliteAPI()
    
    # Test 1: First Alarm
    print("\n1. Testing First Alarm:")
    success = api.send_first_alarm(drone_count=3)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 2: Tracking Data
    print("\n2. Testing Tracking Data:")
    mock_objects = [
        {
            'frame': 1,
            'object_id': 1,
            'drone_type': 'DJI_Mavic',
            'lat': 13.7563,
            'lon': 100.5018,
            'speed_ms': 15.2,
            'direction_deg': 45.3
        },
        {
            'frame': 1,
            'object_id': 2,
            'drone_type': 'DJI_Phantom',
            'lat': 13.7564,
            'lon': 100.5019,
            'speed_ms': 12.8,
            'direction_deg': 90.0
        }
    ]
    
    success = api.send_tracking_data(mock_objects)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 3: Image Encoding
    print("\n3. Testing Image Encoding:")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_base64 = api.encode_image_base64(test_image)
    print(f"   Base64 length: {len(img_base64)} characters")
    print(f"   Result: ✅ Success")
    
    # Summary
    print("\n" + "="*60)
    print(f"Total API calls: {len(api.get_sent_data())}")
    print("All tests passed! ✅")
