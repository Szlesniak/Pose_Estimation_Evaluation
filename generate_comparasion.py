import cv2
import numpy as np
import os

path_orig = '/home/moris/demo/demo_group.mp4'
path_rtm = 'outputs/demo_results/rtmpose_clean.mp4'
path_vit = 'outputs/demo_results/vitpose_clean.mp4'
path_yolo = 'outputs/demo_results/yolo_clean.mp4'

for path in [path_orig, path_rtm, path_vit, path_yolo]:
    if not os.path.exists(path):
        print(f"file not found at: {path}")
        exit()

# 2. Open all four videos
cap_orig = cv2.VideoCapture(path_orig)
cap_rtm = cv2.VideoCapture(path_rtm)
cap_vit = cv2.VideoCapture(path_vit)
cap_yolo = cv2.VideoCapture(path_yolo)

# Get video properties from original
fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. Setup the Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/demo_results/COLOR_OVERLAY.mp4', 
                      fourcc, fps, (width, height))

print("recoloring...")
frame_count = 0

COLOR_RTM = (0, 0, 255)   
COLOR_VIT = (0, 255, 0)   
COLOR_YOLO = (255, 0, 0)  

while True:
    ret0, frame_orig = cap_orig.read()
    ret1, frame_rtm = cap_rtm.read()
    ret2, frame_vit = cap_vit.read()
    ret3, frame_yolo = cap_yolo.read()

    if not (ret0 and ret1 and ret2 and ret3):
        break

    frame_rtm = cv2.resize(frame_rtm, (width, height))
    frame_vit = cv2.resize(frame_vit, (width, height))
    frame_yolo = cv2.resize(frame_yolo, (width, height))

    
    def extract_and_color(frame_ai, color):
        diff = cv2.absdiff(frame_ai, frame_orig)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        
        color_canvas = np.zeros_like(frame_orig)
        color_canvas[:] = color
        
        return cv2.bitwise_and(color_canvas, color_canvas, mask=mask)

    skel_rtm = extract_and_color(frame_rtm, COLOR_RTM)
    skel_vit = extract_and_color(frame_vit, COLOR_VIT)
    skel_yolo = extract_and_color(frame_yolo, COLOR_YOLO)

    combined_skels = cv2.add(cv2.add(skel_rtm, skel_vit), skel_yolo)

    dark_background = cv2.addWeighted(frame_orig, 0.35, np.zeros_like(frame_orig), 0.65, 0)
    
    final_frame = cv2.add(dark_background, combined_skels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = height / 720.0 * 0.8
    thick = max(1, int(scale * 2))
    
    cv2.putText(final_frame, "RTMPose", (40, int(60 * (height/720.0))), font, scale, COLOR_RTM, thick, cv2.LINE_AA)
    cv2.putText(final_frame, "ViTPose", (40, int(100 * (height/720.0))), font, scale, COLOR_VIT, thick, cv2.LINE_AA)
    cv2.putText(final_frame, "YOLO11m", (40, int(140 * (height/720.0))), font, scale, COLOR_YOLO, thick, cv2.LINE_AA)

    out.write(final_frame)
    frame_count += 1

cap_orig.release()
cap_rtm.release()
cap_vit.release()
cap_yolo.release()
out.release()

print(f"{frame_count} frames processed")
print("saved to: outputs/demo_results/COLOR_OVERLAY.mp4")
