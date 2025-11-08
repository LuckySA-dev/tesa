# TESA Drone Detection & Tracking Competition

โปรเจคสำหรับการแข่งขัน/การบ้านตรวจจับและติดตามโดรน มีทั้งหมด 3 ปัญหา:

1. **Problem 1**: Drone Detection with OBB (Oriented Bounding Box)
2. **Problem 2**: Drone Localization (คำนวณพิกัดจริง lat/lon/alt)
3. **Problem 3**: Drone Tracking (ติดตามโดรนในวิดีโอ)

---

## 📁 โครงสร้างไฟล์

```
tesa/
├── images/                      # ภาพสำหรับ Problem 1 & 2
│   ├── drones.jpg
│   └── ...
├── videos/                      # วิดีโอสำหรับ Problem 3 (ถ้ามี)
├── problem1_detection.py        # Problem 1: Detection
├── problem2_localization.py     # Problem 2: Localization
├── problem3_tracking.py         # Problem 3: Tracking
├── utils.py                     # Helper functions
├── visualize.py                 # Visualization tools
├── requirements.txt             # Dependencies
├── image_meta.csv              # Metadata สำหรับ Problem 2
└── README.md                   # เอกสารนี้
```

---

## 🚀 การติดตั้ง

### 1. Clone หรือ Download โปรเจค

```bash
cd c:\Users\User\Desktop\Coding\tesa
```

### 2. สร้าง Virtual Environment (แนะนำ)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. ติดตั้ง Dependencies

```powershell
pip install -r requirements.txt
```

### 4. ดาวน์โหลด YOLO Model (ครั้งแรกจะดาวน์โหลดอัตโนมัติ)

โมเดลจะถูกดาวน์โหลดอัตโนมัติเมื่อรันครั้งแรก หรือดาวน์โหลดเองได้:

```python
from ultralytics import YOLO
model = YOLO('yolov8m-obb.pt')  # จะดาวน์โหลดอัตโนมัติ
```

---

## 📖 วิธีใช้งาน

### Problem 1: Drone Detection

ตรวจจับโดรนในภาพนิ่ง → Output: `p1_detection_obb.csv`

```powershell
# พื้นฐาน
python problem1_detection.py --images images --output p1_detection_obb.csv

# ปรับแต่ง
python problem1_detection.py `
    --images images/p1_images `
    --output p1_detection_obb.csv `
    --model yolov8m-obb.pt `
    --conf 0.25 `
    --device cuda
```

**Parameters:**
- `--images`: โฟลเดอร์ภาพ
- `--output`: ไฟล์ CSV ผลลัพธ์
- `--model`: โมเดล YOLO-OBB (n/s/m/l/x)
- `--conf`: Confidence threshold (0-1)
- `--device`: `cuda` หรือ `cpu`

**Output Format:**
```csv
img_file,center_x,center_y,w,h,theta
img_0001.jpg,0.50,0.45,0.20,0.15,15.0
```

---

### Problem 2: Drone Localization

คำนวณพิกัดโดรนในโลกจริง → Output: `p2_localization.csv`

**ต้องมี:** 
- ผลลัพธ์จาก Problem 1 (`p1_detection_obb.csv`)
- ไฟล์ metadata (`image_meta.csv`)

```powershell
# พื้นฐาน
python problem2_localization.py `
    --detection p1_detection_obb.csv `
    --metadata image_meta.csv `
    --output p2_localization.csv

# ปรับแต่ง
python problem2_localization.py `
    --detection p1_detection_obb.csv `
    --metadata image_meta.csv `
    --output p2_localization.csv `
    --pitch -30.0 `
    --yaw 0.0 `
    --drone-alt 100.0 `
    --fov-h 60.0 `
    --fov-v 45.0
```

**Parameters:**
- `--detection`: CSV จาก Problem 1
- `--metadata`: CSV ข้อมูลกล้อง
- `--pitch`: มุมเงยกล้อง (degrees)
- `--yaw`: ทิศทางกล้อง (0=เหนือ, 90=ตะวันออก)
- `--drone-alt`: ประมาณการ altitude โดรน (meters)
- `--fov-h/v`: Field of View กล้อง (degrees)

**Metadata Format (`image_meta.csv`):**
```csv
img_file,img_lat,img_lon,img_alt
img_0001.jpg,13.123456,100.987654,50.0
```

**Output Format:**
```csv
img_file,center_x,center_y,w,h,theta,drone_lat,drone_lon,drone_alt
img_0001.jpg,0.50,0.45,0.20,0.15,10.0,13.123800,100.987900,120.0
```

---

### Problem 3: Drone Tracking

ติดตามโดรนในวิดีโอ → Output: `p3_tracking_obb.csv`

```powershell
# วิดีโอเดียว
python problem3_tracking.py `
    --video videos/video_01.mp4 `
    --output p3_tracking_obb.csv

# หลายวิดีโอ
python problem3_tracking.py `
    --videos videos/ `
    --output p3_tracking_obb.csv `
    --model yolov8m-obb.pt `
    --conf 0.25 `
    --track-thresh 0.5
```

**Parameters:**
- `--video`: ไฟล์วิดีโอเดียว
- `--videos`: โฟลเดอร์วิดีโอ
- `--model`: โมเดล YOLO-OBB
- `--conf`: Detection confidence
- `--track-thresh`: Tracking threshold
- `--skip-frames`: Skip frames (ประมวลผลเร็วขึ้น)

**Output Format:**
```csv
video_id,frame_id,track_id,center_x,center_y,w,h,theta
video_01,0,1,0.52,0.40,0.18,0.14,5.0
video_01,0,2,0.30,0.55,0.20,0.16,-3.0
video_01,1,1,0.53,0.41,0.18,0.14,5.5
```

---

## 🎨 Visualization

แสดงผลลัพธ์เป็นภาพ/วิดีโอ

```powershell
# Problem 1: Detection
python visualize.py `
    --problem 1 `
    --image images/img_0001.jpg `
    --csv p1_detection_obb.csv `
    --output output/vis_detection.jpg

# Problem 2: Localization
python visualize.py `
    --problem 2 `
    --image images/img_0001.jpg `
    --csv p2_localization.csv `
    --output output/vis_localization.jpg

# Problem 3: Tracking
python visualize.py `
    --problem 3 `
    --video videos/video_01.mp4 `
    --csv p3_tracking_obb.csv `
    --output output/vis_tracking.mp4
```

---

## 🔧 Utilities

### สร้าง Sample Metadata

```python
from utils import create_sample_metadata

df = create_sample_metadata(
    image_folder='images',
    output_csv='image_meta.csv',
    base_lat=13.7563,  # Bangkok
    base_lon=100.5018,
    base_alt=50.0
)
```

### คำนวณ IoU ของ OBB

```python
from utils import compute_iou_obb

obb1 = {'center_x': 0.5, 'center_y': 0.5, 'w': 0.2, 'h': 0.1, 'theta': 0}
obb2 = {'center_x': 0.55, 'center_y': 0.5, 'w': 0.2, 'h': 0.1, 'theta': 10}

iou = compute_iou_obb(obb1, obb2, method='bbox')
print(f"IoU: {iou:.3f}")
```

---

## 📊 YOLO OBB Format

รูปแบบ Oriented Bounding Box ที่ใช้:

```
center_x, center_y, w, h, theta
```

- **center_x, center_y**: จุดกึ่งกลาง (normalized 0-1)
- **w, h**: ความกว้าง/สูง (normalized 0-1)
- **theta**: มุมหมุน (degrees, -90 ถึง +90)

### ตัวอย่าง:
```
0.50, 0.45, 0.20, 0.15, 15.0
```
= โดรนอยู่กึ่งกลางภาพ, กว้าง 20%, สูง 15%, เอียง 15° ทวนเข็มนาฬิกา

---

## ⚙️ การปรับแต่ง

### เลือกโมเดล YOLO

| Model | ขนาด | ความเร็ว | ความแม่นยำ |
|-------|------|---------|-----------|
| yolov8n-obb.pt | เล็กสุด | เร็วสุด | ⭐⭐ |
| yolov8s-obb.pt | เล็ก | เร็ว | ⭐⭐⭐ |
| yolov8m-obb.pt | กลาง | ปานกลาง | ⭐⭐⭐⭐ (แนะนำ) |
| yolov8l-obb.pt | ใหญ่ | ช้า | ⭐⭐⭐⭐⭐ |
| yolov8x-obb.pt | ใหญ่สุด | ช้าสุด | ⭐⭐⭐⭐⭐⭐ |

### Fine-tune โมเดล

ถ้ามี dataset โดรนของตัวเอง:

```python
from ultralytics import YOLO

# โหลดโมเดล pretrained
model = YOLO('yolov8m-obb.pt')

# Train
model.train(
    data='drone_dataset.yaml',  # config file
    epochs=100,
    imgsz=640,
    batch=16,
    name='drone_obb'
)
```

---

## 🐛 Troubleshooting

### Error: CUDA out of memory
- ใช้โมเดลเล็กลง (n หรือ s)
- หรือใช้ CPU: `--device cpu`

### โดรนตรวจไม่เจอ
- ลด `--conf` threshold (เช่น 0.15)
- ใช้โมเดลใหญ่ขึ้น (m → l → x)
- Fine-tune ด้วย dataset โดรน

### Tracking ไม่ติด
- เพิ่ม `--track-thresh` (เช่น 0.7)
- ลด `--conf` เพื่อให้ detect ได้มากขึ้น

---

## 📦 การส่งงาน

ส่งไฟล์ ZIP ที่มี:

1. **Source code** (ไฟล์ .py ทั้งหมด)
2. **requirements.txt**
3. **README.md** (คำแนะนำวิธีรัน)
4. **ผลลัพธ์ CSV** (p1, p2, p3)

**ห้าม** ใส่:
- โฟลเดอร์ภาพ/วิดีโอ
- โมเดล (.pt)
- Virtual environment

```powershell
# สร้าง ZIP
Compress-Archive -Path problem*.py,utils.py,visualize.py,requirements.txt,README.md,*.csv -DestinationPath submission.zip
```

---

## 📚 เอกสารเพิ่มเติม

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO-OBB Guide](https://docs.ultralytics.com/tasks/obb/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

---

## 👨‍💻 Author

TESA Drone Detection Competition Project

---

## 📄 License

MIT License - ใช้ได้เลย! 🚀
