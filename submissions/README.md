# TESA Defence - Submissions Folder

**Generated:** November 8, 2025  
**Status:** ✅ Ready for Competition

---

## 📁 Submission Files

### 🎯 Final Submissions (For Competition)

| File | Purpose | Deadline | Format |
|------|---------|----------|--------|
| **p2_localization_final.csv** | Problem 1 & 2 | 11-12 พ.ย. 18:00 | `frame_id, object_id, direction, distance, height` |
| **submission.csv** | Integration | 13 พ.ย. 20:00 | `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred` |

### 📊 Alternative/Supporting Files

| File | Description |
|------|-------------|
| `p1_detection_obb.csv` | Problem 1 with normalized YOLO OBB format |
| `submission_normalized.csv` | Integration with normalized coordinates |
| `problem1_bytetrack.csv` | Original ByteTrack output (pixels) |
| `training_dataset.csv` | Dataset used for XGBoost training (248 samples) |

### 🔬 Testing/Analysis Files

| File | Description |
|------|-------------|
| `problem1_conf05.csv` | Confidence 0.5 testing |
| `problem1_conf055.csv` | Confidence 0.55 testing (optimal) |
| `problem1_conf06.csv` | Confidence 0.6 testing |
| `problem2_predictions.csv` | Regression predictions |
| `ground_truth_mock.csv` | Mock ground truth for testing |
| `test_api_integration.csv` | API integration test |

---

## 📋 File Details

### 1. p2_localization_final.csv (248 records)
**Format:** `frame_id, object_id, direction, distance, height`

```csv
frame_id, object_id, direction, distance, height
0, 1, 34.2, 54.7, 4.9
0, 2, 324.0, 50.0, 1.2
```

**Specifications:**
- ✅ direction: 28.3° - 343.9° (compass degrees)
- ✅ distance: 50.0 - 74.5 m
- ✅ height: -1.6 - 9.3 m (calculated from elevation)

---

### 2. submission.csv (248 records)
**Format:** `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred`

```csv
video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
video_01.mp4, 0, 1803, 565, 54.7, 34.2, 5.1
```

**Specifications:**
- ✅ cx, cy: PIXELS (169-1829, 472-712)
- ✅ range_m: 50.0 - 74.5 m
- ✅ azimuth: -37.5° - 35.3°
- ✅ elevation: -1.3° - 9.2°

---

### 3. p1_detection_obb.csv (248 records)
**Format:** `frame_id, object_id, center_x, center_y, w, h`

```csv
frame_id, object_id, center_x, center_y, w, h
0, 1, 0.8806, 0.4142, 0.0747, 0.1085
```

**Specifications:**
- ✅ Normalized coordinates (0-1)
- ✅ YOLO OBB format compliant
- ✅ center_x: 0.0825 - 0.8931
- ✅ center_y: 0.3464 - 0.5224

---

## 🔄 Conversion Pipeline

```
video_01.mp4
    ↓
YOLO-OBB Detection (conf=0.55)
    ↓
ByteTrack Tracking
    ↓
problem1_bytetrack.csv (pixels)
    ↓
[Normalize] → p1_detection_obb.csv
    ↓
XGBoost Regression (training_dataset.csv)
    ↓
submission.csv (pixels) ← INTEGRATION FORMAT
    ↓
[Convert] → p2_localization_final.csv (direction/distance/height)
```

---

## 📊 Statistics

### Detection & Tracking:
- **Frames processed:** 120
- **Total predictions:** 248
- **Unique tracks:** 3
- **Processing speed:** 6.6 FPS
- **Confidence threshold:** 0.55 (optimized)

### Regression Performance:
- **Range MAE:** 0.954 m
- **Azimuth MAE:** 1.430°
- **Elevation MAE:** 0.509°
- **Overall R²:** 0.999

---

## ✅ Validation

All files validated with `check_compliance.py`:
- ✅ Problem 1: PASS
- ✅ Problem 2: PASS
- ✅ Integration: PASS
- ✅ Format compliance: 100%

---

## 📅 Submission Checklist

- [ ] **11 พ.ย. 18:00** - Submit `p2_localization_final.csv`
- [ ] **12 พ.ย. 18:00** - Submit `p2_localization_final.csv`
- [ ] **13 พ.ย. 20:00** - Submit `submission.csv`

**Current Date:** November 8, 2025  
**Days Remaining:** 3-5 days  
**Status:** 🎉 ALL FILES READY
