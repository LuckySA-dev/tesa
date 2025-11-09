# 🚀 Quick Start - 5 Minutes

**TESA Drone Detection System - Fast Setup Guide**

---

## ⚡ 1. Install (1 minute)

```bash
pip install -r requirements.txt
```

---

## 🎯 2. Run (3 minutes)

### **Problem 1: Detection**
```bash
python problem1_competition.py --video videos/video_01.mp4
```
**Output:** `submissions/p1_detection_obb.csv` (248 detections, ~22s)

### **Problem 2: Localization** 
⚠️ **Must run Problem 1 first!**
```bash
python problem2_inference.py --detections submissions/p1_detection_obb.csv --video videos/video_01.mp4
```
**Output:** `submissions/predictions.csv` (GPS regression)

### **Problem 3: Full Pipeline**
```bash
python problem3_integration.py --video videos/video_01.mp4
```
**Output:** `submissions/submission.csv` (complete results)

---

## ✅ 3. Validate (1 minute)

```bash
python check_compliance.py
```
**Result:** All submissions validated ✅

---

## 🥧 Raspberry Pi 5

```bash
# Optimized version
python problem1_raspberry_pi.py --video videos/video_01.mp4 --skip 2
```
**Performance:** ~45-75s (vs 22s desktop)

---

## 📊 Web Dashboard

```bash
pip install streamlit plotly
streamlit run dashboard_streamlit.py
```
**Opens:** http://localhost:8501 (beautiful UI!)

---

## 🎯 Most Used Commands

```bash
# Basic detection (recommended)
python problem1_competition.py --video videos/video_01.mp4

# Lower confidence (more detections)
python problem1_competition.py --video videos/video_01.mp4 --conf 0.4

# Raspberry Pi (faster)
python problem1_raspberry_pi.py --video videos/video_01.mp4 --skip 2

# Validate all
python check_compliance.py

# Web UI
streamlit run dashboard_streamlit.py
```

---

## 📁 Expected Results

```
submissions/
├── p1_detection_obb.csv         ✅ 248 detections
├── p2_localization_final.csv    ✅ GPS coordinates
└── submission.csv               ✅ Complete pipeline
```

---

## ⚠️ Troubleshooting

**Module not found?**
```bash
pip install -r requirements.txt
```

**CUDA error?**
```bash
# Force CPU
set CUDA_VISIBLE_DEVICES=
```

**Low performance?**
```bash
# Use skip frames
python problem1_raspberry_pi.py --video videos/video_01.mp4 --skip 2
```

---

## 📚 Need More Help?

**Full guide:** `HOW_TO_RUN.md`  
**Documentation:** `reports/` directory  
**Organization:** `PROJECT_ORGANIZATION.md`

---

## 🏆 You're Ready!

```bash
python problem1_competition.py --video videos/video_01.mp4
```

**Good luck! 🎯**
