# 📁 TESA Defence - Project Organization

**Date:** November 8, 2025  
**Status:** ✅ Fully Organized + Dataset Agnostic Ready

---

## 🚀 **NEW:** Dataset Agnostic System

ระบบได้รับการอัพเกรดเป็น **Dataset Agnostic** แล้ว!

### ✅ สิ่งที่เปลี่ยน:
- 🎯 Auto-detect video resolution, FPS
- 🎯 Adaptive ByteTrack with dynamic FPS  
- 🎯 Resolution-aware inference
- 🎯 Auto-tune confidence threshold utility

### 📖 อ่านเพิ่มเติม:
- **Full Details:** `reports/DATASET_AGNOSTIC.md`
- **Quick Start:** `reports/QUICK_REFERENCE.md`

---

## 📂 Folder Structure

```
tesa/
├── submissions/          ← 📊 All CSV files (19 files)
│   ├── p1_detection_obb.csv          ⭐ Problem 1 submission
│   ├── p2_localization_final.csv     ⭐ Problem 2 submission
│   ├── submission.csv                ⭐ Integration submission
│   ├── submission_normalized.csv     Alternative version
│   ├── training_dataset.csv          Training data
│   └── ... (14 more CSV files)
│
├── reports/              ← 📄 All MD documentation (14 files)
│   ├── FINAL_STATUS.md               ⭐ Project status
│   ├── DATASET_AGNOSTIC.md           🚀 NEW: Dataset agnostic guide
│   ├── QUICK_REFERENCE.md            📖 NEW: Quick start guide
│   ├── FORMAT_CLARIFICATION.md       🚨 Critical info
│   ├── GAP_ANALYSIS_FINAL.md         Compliance check
│   ├── validation_report.md          Performance metrics
│   ├── INDEX.md                      Report index
│   └── ... (7 more MD files)
│
├── models/               ← 🧠 XGBoost models
│   ├── range_m_xgboost.pkl
│   ├── azimuth_deg_xgboost.pkl
│   └── elevation_deg_xgboost.pkl
│
├── videos/               ← 🎬 Video files
│   └── video_01.mp4
│
├── output/               ← 🎥 Generated videos
│   └── complete_system.mp4
│
├── logs/                 ← 📝 Log files
│   └── test_tracking.csv
│
└── *.py                  ← 💻 Python scripts (21 files)
    ├── problem1_competition.py       ⭐ Problem 1 system (dataset agnostic)
    ├── problem2_train.py             ⭐ Model training
    ├── problem2_inference.py         ⭐ Inference (auto-detect dimensions)
    ├── problem3_integration.py       ⭐ Complete pipeline (adaptive)
    ├── check_compliance.py           ✅ Validation
    ├── auto_tune_confidence.py       🆕 Auto-tune threshold
    ├── fix_problem2_format.py        🔧 Format converter
    ├── fix_obb_normalization.py      🔧 Normalizer
    └── ... (more scripts)
```

---

## 📊 File Organization Summary

### submissions/ folder (CSV files)
**Purpose:** All dataset and output CSV files

**Final Submissions:**
- `p1_detection_obb.csv` (248 records) - Problem 1
- `p2_localization_final.csv` (248 records) - Problem 2
- `submission.csv` (248 records) - Integration

**Supporting Files:**
- Training data: `training_dataset.csv`
- Analysis files: `problem1_*.csv`, `problem2_*.csv`
- Test files: `ground_truth_mock.csv`, `test_api_integration.csv`

**Total:** 19 CSV files

---

### reports/ folder (Documentation)
**Purpose:** All markdown documentation and reports

**Must Read:**
- `FINAL_STATUS.md` - Current project status
- `FORMAT_CLARIFICATION.md` - Format requirements
- `validation_report.md` - Performance metrics

**Analysis:**
- `GAP_ANALYSIS_FINAL.md` - Compliance analysis
- `COMPARISON_REPORT.md` - Tracker comparison
- `IMPROVEMENTS_FINAL.md` - Performance improvements

**Documentation:**
- `README.md` - Main documentation
- `README_SYSTEM.md` - Architecture
- `INDEX.md` - Report index

**Total:** 12 MD files

---

## 🔧 Updated Scripts

The following scripts have been updated with new paths:

### Core Scripts:
✅ `check_compliance.py` - All paths updated to `submissions/`  
✅ `problem3_integration.py` - Output to `submissions/submission.csv`  
✅ `fix_problem2_format.py` - Input/output from `submissions/`  
✅ `problem1_competition.py` - Output to `submissions/`  
✅ `problem2_train.py` - Dataset from `submissions/`

### Usage Examples:
```bash
# Compliance check (automatically uses submissions/ folder)
python check_compliance.py

# Generate integration output
python problem3_integration.py --video videos/video_01.mp4 --output submissions/submission.csv

# Convert Problem 2 format
python fix_problem2_format.py --input submissions/submission.csv --output submissions/p2_localization_final.csv

# Train models
python problem2_train.py --dataset submissions/training_dataset.csv
```

---

## 📅 Submission Files Location

All submission files are in `submissions/` folder:

| Deadline | File Path | Format |
|----------|-----------|--------|
| 11 พ.ย. 18:00 | `submissions/p2_localization_final.csv` | direction, distance, height |
| 12 พ.ย. 18:00 | `submissions/p2_localization_final.csv` | direction, distance, height |
| 13 พ.ย. 20:00 | `submissions/submission.csv` | cx, cy, predictions (pixels) |

---

## ✅ Benefits of New Organization

### Before:
```
tesa/
├── p1_detection_obb.csv
├── p2_localization_final.csv
├── submission.csv
├── FINAL_STATUS.md
├── FORMAT_CLARIFICATION.md
├── ... (31 mixed files in root)
└── problem1_competition.py
```

### After:
```
tesa/
├── submissions/      ← All CSV files
├── reports/          ← All MD files
├── models/           ← Model files
└── *.py              ← Scripts only in root
```

**Improvements:**
✅ Clear separation of data/docs/code  
✅ Easy to find submission files  
✅ Better for version control  
✅ Professional structure  
✅ Easier to navigate

---

## 🎯 Quick Reference

**Need submission files?**  
→ Look in `submissions/` folder

**Need documentation?**  
→ Look in `reports/` folder

**Need to run pipeline?**  
→ Scripts in root (`.py` files)

**Need models?**  
→ Look in `models/` folder

---

## 📝 Notes

1. All Python scripts automatically use correct paths
2. `check_compliance.py` validates files in `submissions/`
3. Reports folder includes index: `reports/INDEX.md`
4. Submissions folder includes guide: `submissions/README.md`

---

**Organization Complete!** ✅  
**All paths updated!** ✅  
**Ready for submission!** 🎉
