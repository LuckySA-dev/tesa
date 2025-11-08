# TESA Defence - Gap Analysis Report (Based on New Requirements)

**Date:** November 8, 2025  
**Analysis Based On:** ### TESA Defence.txt (Updated Requirements)

---

## ✅ สิ่งที่ทำถูกต้องตามเกณฑ์ใหม่

### 1. Problem 1: Drone Detection (11 พ.ย.)
**เกณฑ์:** object detection + tracking → output: `frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h`

✅ **ทำถูกแล้ว:**
- File: `problem1_competition.py` 
- Output format: ✅ `frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h`
- YOLO-OBB detection: ✅ Implemented
- ByteTrack tracking: ✅ Implemented (ตามโจทย์แนะนำ)
- Confidence threshold: ✅ 0.55 (optimized)
- Output file: ✅ `problem1_bytetrack.csv` (248 detections, 3 objects)

**Status:** ✅ PASS - ตรงตามโจทย์ Problem 1

---

### 2. Problem 2: Model Predict Location (12 พ.ย.)
**เกณฑ์:** Model ทำนาย → output: `frame_id, object_id, direction, distance, height`

⚠️ **ปัญหาพบ:**
- Current output: `range_m, azimuth_deg, elevation_deg` 
- Required output: `direction, distance, height`
- **Mismatch คอลัมน์:**
  - ✅ `range_m` = `distance` (ระยะทาง, เมตร)
  - ❌ `azimuth_deg` ≠ `direction` (ควรเป็น direction เช่น "N", "NE", "E" หรือเป็นองศา 0-360°?)
  - ❌ `elevation_deg` ≠ `height` (elevation เป็นมุม, height เป็นความสูง เมตร)

**Status:** ❌ PARTIAL FAIL - Format ไม่ตรงตามโจทย์ Problem 2

---

### 3. Integration (13 พ.ย. deadline 20:00)
**เกณฑ์จากไฟล์ใหม่:** `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred`

✅ **ทำถูกแล้ว:**
- File: `problem3_integration.py`
- Output: `submission.csv` ✅ Format ตรงตามเกณฑ์ Integration
- Columns: ✅ `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred`
- Sample data: ✅ 248 predictions from video_01.mp4

**Status:** ✅ PASS - ตรงตามโจทย์ Integration (13 พ.ย.)

---

## 🚨 จุดผิดพลาดสำคัญ

### ❌ Problem 2 Output Format Mismatch

**โจทย์ 12 พ.ย. ต้องการ:**
```
frame_id, object_id, direction, distance, height
```

**ระบบปัจจุบันให้:**
```
frame_id, object_id, range_m, azimuth_deg, elevation_deg
```

### 🔍 การแปลงที่ต้องทำ:

1. **`direction` (ทิศทาง)**
   - Input: `azimuth_deg` (-180° ถึง +180° หรือ 0° ถึง 360°)
   - Output ที่ควรเป็น:
     - แบบที่ 1: องศา 0-360° (0=North, 90=East, 180=South, 270=West)
     - แบบที่ 2: ตัวอักษร "N", "NE", "E", "SE", "S", "SW", "W", "NW"
   
2. **`distance` (ระยะทาง)**
   - ✅ `range_m` → `distance` (เหมือนกัน, แค่เปลี่ยนชื่อ)

3. **`height` (ความสูง)**
   - Input: `elevation_deg` (มุมก้ม/เงย เป็นองศา)
   - ⚠️ **ต้องแปลงเป็นความสูงจริง (เมตร)**
   - สูตร: `height = distance × tan(elevation_deg)`
   - ตัวอย่าง: 
     - distance=50m, elevation=10° → height = 50 × tan(10°) ≈ 8.8m

---

## 📋 YOLO OBB Format Compliance

**เกณฑ์:** `center_x, center_y, w, h, theta` (normalized 0-1 ยกเว้น theta)

### ปัญหาที่พบ:

❌ **ระบบปัจจุบันใช้พิกเซลโดยตรง ไม่ได้ normalize!**

```python
# Current output (problem1_competition.py):
cx, cy = int(xywhr[0]), int(xywhr[1])  # พิกเซล (เช่น 1803, 565)
w, h = int(xywhr[2]), int(xywhr[3])    # พิกเซล

# Should be normalized (0-1):
cx_norm = cx / image_width   # เช่น 1803/2048 = 0.880
cy_norm = cy / image_height  # เช่น 565/1364 = 0.414
w_norm = w / image_width
h_norm = h / image_height
```

**ผลกระทบ:**
- ❌ Problem 1 output ไม่ได้ใช้ normalized coordinates
- ❌ Training dataset features (cx, cy, w, h) ใช้พิกเซลแทนที่จะเป็น 0-1
- ⚠️ XGBoost models train จาก pixel values แทน normalized values

---

## 📊 Summary: Compliance Check

| Requirement | Expected Format | Current Format | Status |
|-------------|----------------|----------------|--------|
| **Problem 1 (11 พ.ย.)** | `frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h` | ✅ Matches | ✅ PASS |
| **Problem 2 (12 พ.ย.)** | `frame_id, object_id, direction, distance, height` | ❌ `frame_id, object_id, range_m, azimuth_deg, elevation_deg` | ❌ FAIL |
| **Integration (13 พ.ย.)** | `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred` | ✅ Matches | ✅ PASS |
| **YOLO OBB Format** | Normalized 0-1 for cx, cy, w, h | ❌ Pixel values | ❌ FAIL |

---

## 🔧 Action Items ที่ต้องแก้ไข

### Priority 1 (CRITICAL - ส่งผล 12 พ.ย.):
1. ✅ **แก้ Problem 2 output format:**
   - เพิ่มฟังก์ชันแปลง `azimuth_deg` → `direction` (0-360° or compass)
   - เพิ่มฟังก์ชันแปลง `elevation_deg + range_m` → `height` (เมตร)
   - เปลี่ยน column names: `range_m` → `distance`

### Priority 2 (RECOMMENDED):
2. ⚠️ **แก้ YOLO OBB Format:**
   - Normalize cx, cy, w, h เป็น 0-1 ใน detection phase
   - Re-train XGBoost models ด้วย normalized features
   - Update problem1_competition.py output

### Priority 3 (OPTIONAL):
3. 📝 **Document clarifications:**
   - ระบุว่า `direction` หมายถึงอะไร (องศา หรือ compass?)
   - ระบุสมมติฐานการคำนวณ height (ใช้ระนาบอ้างอิงใด)

---

## 📐 Mathematical Formulas Needed

### 1. Azimuth → Direction (0-360°)
```python
def azimuth_to_direction(azimuth_deg):
    """
    Convert azimuth (-180 to +180) to direction (0-360)
    0° = North, 90° = East, 180° = South, 270° = West
    """
    direction = (azimuth_deg + 360) % 360
    return direction
```

### 2. Elevation + Range → Height
```python
import math

def calculate_height(range_m, elevation_deg):
    """
    Calculate drone height from range and elevation angle
    
    Args:
        range_m: Distance to drone (meters)
        elevation_deg: Elevation angle (degrees)
    
    Returns:
        height_m: Vertical height above camera (meters)
    """
    elevation_rad = math.radians(elevation_deg)
    height_m = range_m * math.sin(elevation_rad)
    return height_m

# Example:
# range=50m, elevation=10° → height = 50 × sin(10°) ≈ 8.68m
```

### 3. Normalize Coordinates
```python
def normalize_bbox(cx, cy, w, h, image_width, image_height):
    """Normalize bounding box to 0-1 range"""
    cx_norm = cx / image_width
    cy_norm = cy / image_height
    w_norm = w / image_width
    h_norm = h / image_height
    return cx_norm, cy_norm, w_norm, h_norm
```

---

## ✅ Recommended Solution Order

1. **Today (Nov 8):**
   - ✅ Create `problem2_localization_fixed.py` with correct format
   - ✅ Add conversion functions (azimuth→direction, elevation→height)
   - ✅ Test with existing predictions

2. **Nov 9-10:**
   - ⚠️ Consider re-training models with normalized coordinates
   - 📝 Update documentation

3. **Nov 11-12:**
   - ✅ Submit Problem 1 (already compliant)
   - ✅ Submit Problem 2 (with fixes)

4. **Nov 13 (deadline 20:00):**
   - ✅ Submit Integration (already compliant)

---

## 📝 Notes

- **Good news:** Integration format (13 พ.ย.) ถูกต้องแล้ว! 
- **Issue:** Problem 2 format (12 พ.ย.) ต้องแก้
- **Optional:** YOLO OBB normalization (ไม่บังคับแต่แนะนำ)

Current overall compliance: **2/3 problems correct** (66.7%)  
After fixing Problem 2: **3/3 problems correct** (100%)
