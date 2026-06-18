# RTMPose / ViTPose / YOLO

A complete set of commands and scripts used to evaluate and demonstrate pose estimation models on the COCO val2017 dataset in a WSL2 + RTX 3060 Ti environment.

## 1. Environment Preparation

### 1.1 WSL2 - GPU Verification
```bash
nvidia-smi
```

### 1.2 Miniconda
```bash
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
```
*Close and reopen the terminal.*

### 1.3 Creating Conda Environment
```bash
conda create -n rtmpose python=3.9 -y
conda activate rtmpose
```

### 1.4 PyTorch 2.4.0 + CUDA 12.1 (via pip)
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
**Verification:**
```bash
$CONDA_PREFIX/bin/python -c "import torch; print(torch.__version__)"
```
*Expected output: `2.4.0+cu121`*

### 1.5 OpenMMLab Stack
```bash
pip install -U openmim
$CONDA_PREFIX/bin/python -m mim install mmengine
$CONDA_PREFIX/bin/python -m mim install "mmcv>=2.0.0"
$CONDA_PREFIX/bin/python -m mim install "mmdet>=3.0.0"
```

### 1.6 MMPose from Source
```bash
git clone [https://github.com/open-mmlab/mmpose.git](https://github.com/open-mmlab/mmpose.git)
cd mmpose
pip install -v -e .
```

### 1.7 Additional Dependencies
```bash
pip install "numpy<2.0.0" "setuptools<70.0.0"
pip install json_tricks cffi pyyaml requests
pip install chumpy --no-build-isolation
pip install fsspec "sympy==1.13.1" networkx
pip install mmpretrain          # required for ViTPose
```

### 1.8 mmcv/mmdet Version Workaround
If `mmdet` raises an `AssertionError` regarding `mmcv==2.2.0`:
```bash
sed -i 's/2.2.0/3.0.0/g' \
  $CONDA_PREFIX/lib/python3.9/site-packages/mmdet/__init__.py
```

### 1.9 Ultralytics (YOLO)
```bash
pip install ultralytics
```

### 1.10 Graphics Libraries (for `--show` demo in WSL)
```bash
sudo apt-get update && sudo apt-get install -y \
  libxcb-cursor0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

---

## 2. COCO Dataset Preparation

### 2.1 Folder Structure
```bash
cd ~/mmpose
mkdir -p data/coco/annotations
mkdir -p data/coco/val2017
mkdir -p data/coco/person_detection_results
```

### 2.2 Downloading Data
```bash
cd data/coco/
wget [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
unzip -o annotations_trainval2017.zip
wget [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
unzip -q -o val2017.zip
cd ~/mmpose
```

### 2.3 Faster R-CNN Bounding Boxes (OpenMMLab baseline)
```bash
wget -O data/coco/person_detection_results/\
  COCO_val2017_detections_AP_H_56_person.json \
  [https://huggingface.co/Prophetetc/cocopose/resolve/main/](https://huggingface.co/Prophetetc/cocopose/resolve/main/)\
  COCO_val2017_detections_AP_H_56_person.json
```

**Final structure:**
```text
mmpose/data/coco/
├── annotations/
│   └── person_keypoints_val2017.json
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── person_detection_results/
    ├── COCO_val2017_detections_AP_H_56_person.json
    └── yolo11m_coco_detections.json  (from step 3.1)
```

---

## 3. Model Evaluation

### 3.1 YOLO Bounding Boxes
We generated bounding boxes using the YOLO11m-pose model, and then feed them to the MMPose models.

**`generate_yolo_boxes.py`** (Run from `~/yolo_workspace`):
```python
import os, json
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO('yolo11m-pose.pt')
img_folder = '/home/moris/mmpose/data/coco/val2017'
results_list = []

for img_name in tqdm(os.listdir(img_folder)):
    if not img_name.endswith('.jpg'):
        continue
    image_id = int(img_name.split('.')[0])
    img_path = os.path.join(img_folder, img_name)
    results = model(img_path, verbose=False)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            if int(classes[i]) == 0:
                x1, y1, x2, y2 = boxes[i]
                results_list.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [float(x1), float(y1),
                             float(x2-x1), float(y2-y1)],
                    'score': float(scores[i])
                })

with open('yolo11m_coco_detections.json', 'w') as f:
    json.dump(results_list, f)
```

**ViTPose-L with YOLO bbox**
`tools/test.py`: [Source Code](https://github.com/open-mmlab/mmpose/blob/main/tools/test.py)
```bash
$CONDA_PREFIX/bin/python tools/test.py \
  configs/body_2d_keypoint/topdown_heatmap/coco/\
  td-hm_ViTPose-large_8xb64-210e_coco-256x192.py \
  [https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/)\
  topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-\
  256x192-53609f55_20230314.pth \
  --cfg-options \
  test_dataloader.dataset.bbox_file=\
  data/coco/person_detection_results/yolo11m_coco_detections.json \
  val_dataloader.batch_size=1 val_dataloader.num_workers=1
```

**RTMPose-t with YOLO bbox**
```bash
$CONDA_PREFIX/bin/python tools/test.py \
  configs/body_2d_keypoint/rtmpose/coco/\
  rtmpose-t_8xb256-420e_coco-256x192.py \
  [https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/)\
  rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-\
  cfc8f33d_20230126.pth \
  --cfg-options \
  test_dataloader.dataset.bbox_file=\
  data/coco/person_detection_results/yolo11m_coco_detections.json \
  val_dataloader.batch_size=1 val_dataloader.num_workers=1
```

### 3.2 YOLO11m-pose (single-stage, standalone)
Run from `~/yolo_workspace`:
```bash
$CONDA_PREFIX/bin/yolo val pose \
  model=yolo11m-pose.pt data=coco-pose.yaml
```
*Note: Ultralytics automatically downloads the COCO-Pose dataset (~18 GB) on the first run.*

---

## 4. Results Location

### 4.1 MMPose (RTMPose, ViTPose)
Results are saved in: `~/mmpose/work_dirs/<config_name>/<timestamp>/`
* `vis_data/scalars.json` ← AP/AR metrics (JSON)
* `<timestamp>.log`       ← Full log + configuration

### 4.2 YOLO
Results are saved in: `~/yolo_workspace/runs/pose/val/`
* `predictions.json`    ← All predictions
* `*curve.png`          ← P/R/F1 plots
* `val_batch*_pred.jpg` ← Visualizations

---

## 5. Additional Metrics

### 5.1 GFLOPs (MMPose)
```bash
$CONDA_PREFIX/bin/python tools/analysis_tools/get_flops.py \
  configs/body_2d_keypoint/rtmpose/coco/\
  rtmpose-t_8xb256-420e_coco-256x192.py

$CONDA_PREFIX/bin/python tools/analysis_tools/get_flops.py \
  configs/body_2d_keypoint/topdown_heatmap/coco/\
  td-hm_ViTPose-large_8xb64-210e_coco-256x192.py
```

### 5.2 GFLOPs (YOLO)
```bash
$CONDA_PREFIX/bin/yolo info model=yolo11m-pose.pt
```

### 5.3 VRAM (GPU resource usage monitoring)
```bash
watch -n 1 nvidia-smi
```

---

## 6. Video Demo

### 6.1 RTMPose-t (with RTMDet as detector)
```bash
$CONDA_PREFIX/bin/python demo/inferencer_demo.py \
  /home/moris/demo/demo_group.mp4 \
  --pose2d configs/body_2d_keypoint/rtmpose/coco/\
  rtmpose-t_8xb256-420e_coco-256x192.py \
  --pose2d-weights [https://download.openmmlab.com/mmpose/v1/](https://download.openmmlab.com/mmpose/v1/)\
  projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_\
  420e-256x192-e613ba3f_20230127.pth \
  --det-model demo/mmdetection_cfg/\
  rtmdet_m_640-8xb32_coco-person.py \
  --det-weights [https://download.openmmlab.com/mmpose/v1/](https://download.openmmlab.com/mmpose/v1/)\
  projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-\
  person-235e8209.pth \
  --vis-out-dir outputs/demo_results/ \
  --thickness 4 --radius 4 --draw-bbox
```
* **Add `--show`** to display a live window (requires graphics libraries from step 1.10).
* **Add `--draw-heatmap`** to view probability heatmaps.

### 6.2 ViTPose-L (with RTMDet as detector)
```bash
$CONDA_PREFIX/bin/python demo/inferencer_demo.py \
  /home/moris/demo/demo_group.mp4 \
  --pose2d configs/body_2d_keypoint/topdown_heatmap/coco/\
  td-hm_ViTPose-large_8xb64-210e_coco-256x192.py \
  --pose2d-weights [https://download.openmmlab.com/mmpose/v1/](https://download.openmmlab.com/mmpose/v1/)\
  body_2d_keypoint/topdown_heatmap/coco/\
  td-hm_ViTPose-large_8xb64-210e_coco-256x192-\
  53609f55_20230314.pth \
  --det-model demo/mmdetection_cfg/\
  rtmdet_m_640-8xb32_coco-person.py \
  --det-weights [https://download.openmmlab.com/mmpose/v1/](https://download.openmmlab.com/mmpose/v1/)\
  projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-\
  person-235e8209.pth \
  --vis-out-dir outputs/demo_results/ \
  --thickness 4 --radius 4 --draw-bbox
```

### 6.3 YOLO11m-pose
```bash
$CONDA_PREFIX/bin/yolo predict \
  model=yolo11m-pose.pt \
  source=/home/moris/demo/demo_group.mp4 \
  save=True
```
*Result saved in: `~/mmpose/runs/pose/predict/` (or `yolo_workspace/runs/`).*
