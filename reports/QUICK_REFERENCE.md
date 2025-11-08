# 📚 Quick Reference - Dataset Agnostic Usage

## 🚀 วิธีใช้งานกับ Video ใหม่

### Option 1: Quick Start (Auto Everything)

```bash
# Step 1: วาง video ใน videos/
cp your_video.mp4 videos/

# Step 2: Auto-tune confidence (แนะนำ)
python auto_tune_confidence.py --video videos/your_video.mp4 --target 2.0

# Output: Optimal confidence: 0.48

# Step 3: Run full pipeline
python problem1_competition.py --video videos/your_video.mp4 --conf 0.48 --output submissions/p1.csv
python problem2_inference.py --detections submissions/p1.csv --video videos/your_video.mp4 --output submissions/p2_temp.csv
python fix_problem2_format.py --input submissions/p2_temp.csv --output submissions/p2_final.csv
python problem3_integration.py --video videos/your_video.mp4 --conf 0.48 --output submissions/submission.csv

# Step 4: Validate
python check_compliance.py
```

---

### Option 2: Manual Control

```bash
# Specify all parameters manually
python problem1_competition.py \
    --video videos/your_video.mp4 \
    --output submissions/p1.csv \
    --conf 0.55

python problem2_inference.py \
    --detections submissions/p1.csv \
    --output submissions/p2.csv \
    --width 1920 \
    --height 1080

python problem3_integration.py \
    --video videos/your_video.mp4 \
    --output submissions/submission.csv \
    --conf 0.55
```

---

## 🎯 Auto-tune Confidence Parameters

### Basic Usage
```bash
python auto_tune_confidence.py --video videos/input.mp4
```

### Advanced Options
```bash
python auto_tune_confidence.py \
    --video videos/input.mp4 \
    --min-conf 0.3 \          # Minimum confidence to test
    --max-conf 0.7 \          # Maximum confidence to test
    --steps 9 \               # Number of steps to test
    --samples 10 \            # Frames to sample
    --target 2.0 \            # Expected drones per frame
    --device auto             # auto, cuda, or cpu
```

---

## 📊 Supported Video Formats

### Resolution
- ✅ 1280x720 (HD)
- ✅ 1920x1080 (Full HD)
- ✅ 2048x1364 (Default)
- ✅ 3840x2160 (4K)
- ✅ Any custom resolution

### Frame Rate
- ✅ 24 FPS
- ✅ 30 FPS (Default)
- ✅ 60 FPS
- ✅ Any custom FPS

### Format
- ✅ .mp4
- ✅ .avi
- ✅ .mov
- ✅ Any OpenCV-supported format

---

## ⚙️ Problem-specific Parameters

### Problem 1: Detection + Tracking
```bash
python problem1_competition.py \
    --video <VIDEO_PATH> \
    --output <OUTPUT_CSV> \
    --conf <CONFIDENCE>       # Use auto-tuned value
```

### Problem 2: Inference
```bash
# Option A: Auto-detect dimensions
python problem2_inference.py \
    --detections <P1_CSV> \
    --output <OUTPUT_CSV> \
    --video <VIDEO_PATH>      # Auto-detect from video

# Option B: Manual dimensions
python problem2_inference.py \
    --detections <P1_CSV> \
    --output <OUTPUT_CSV> \
    --width 1920 \
    --height 1080
```

### Problem 3: Integration
```bash
python problem3_integration.py \
    --video <VIDEO_PATH> \
    --output <OUTPUT_CSV> \
    --conf <CONFIDENCE>       # Use auto-tuned value
```

---

## 🔍 Troubleshooting

### Issue: Too many/few detections
```bash
# Solution: Re-tune confidence
python auto_tune_confidence.py --video your_video.mp4 --target <EXPECTED_DRONES>
```

### Issue: Tracking loses objects
```bash
# Solution: Video has different FPS - system auto-adjusts!
# No action needed, ByteTrack adapts automatically
```

### Issue: Wrong predictions
```bash
# Solution: Check video dimensions
python -c "import cv2; cap = cv2.VideoCapture('video.mp4'); print(f'{int(cap.get(3))}x{int(cap.get(4))}')"
```

---

## ✅ Validation Checklist

```bash
# After processing, always run:
python check_compliance.py

# Should see:
✅ Problem 1: PASS
✅ Problem 2: PASS
✅ Integration: PASS
```

---

## 📁 Expected Output Files

```
submissions/
├── p1_detection_obb.csv          # Problem 1 (11 Nov 18:00)
├── p2_localization_final.csv     # Problem 2 (12 Nov 18:00)
└── submission.csv                # Integration (13 Nov 20:00)
```

---

## 💡 Tips & Best Practices

1. **Always auto-tune confidence first** for new videos
2. **Use --video parameter** in problem2_inference.py for auto-detection
3. **Check output with compliance checker** before submission
4. **Keep confidence between 0.3-0.7** for best results
5. **Sample 10-20 frames** for accurate auto-tuning

---

## 🚨 Common Mistakes to Avoid

❌ **Don't:** Hard-code video dimensions  
✅ **Do:** Use --video parameter or auto-detect

❌ **Don't:** Use same confidence for all videos  
✅ **Do:** Auto-tune for each video

❌ **Don't:** Skip validation  
✅ **Do:** Always run check_compliance.py

❌ **Don't:** Forget to specify output path  
✅ **Do:** Always use --output parameter

---

## 📞 Quick Commands Reference

### Full Pipeline (One-liner)
```bash
python auto_tune_confidence.py --video videos/input.mp4 && \
python problem1_competition.py --video videos/input.mp4 --conf $(cat optimal_conf.txt) --output submissions/p1.csv && \
python problem2_inference.py --detections submissions/p1.csv --video videos/input.mp4 --output submissions/p2_temp.csv && \
python fix_problem2_format.py --input submissions/p2_temp.csv --output submissions/p2.csv && \
python problem3_integration.py --video videos/input.mp4 --conf $(cat optimal_conf.txt) --output submissions/submission.csv && \
python check_compliance.py
```

### Check Video Info
```bash
python -c "import cv2; cap = cv2.VideoCapture('videos/input.mp4'); print(f'Resolution: {int(cap.get(3))}x{int(cap.get(4))} @ {int(cap.get(5))} FPS, Frames: {int(cap.get(7))}')"
```

### Compare Outputs
```bash
python -c "import pandas as pd; df1 = pd.read_csv('old.csv'); df2 = pd.read_csv('new.csv'); print(f'Match: {df1.equals(df2)}')"
```

---

**Last Updated:** November 8, 2025  
**Status:** ✅ Production Ready
