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
