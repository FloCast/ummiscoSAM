import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from patchify import patchify
from ultralytics import YOLOv10
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def calculate_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        raise ValueError("Les deux masques doivent avoir la même forme")
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0.0 if union == 0 else intersection / union

def is_contained_within(box1, box2):
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def patchify_with_border_handling(image, img_patch_size, step):
    H, W = image.shape[:2]
    num_patches_h = (H // step) + 1
    num_patches_w = (W // step) + 1
    pad_h = (num_patches_h * step) - H
    pad_w = (num_patches_w * step) - W
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    return patchify(image_padded, img_patch_size, step=step)

# Configuration
device = "cuda"
size = 1024
step = size
img_patch_size = (size, size, 3)
checkpoint = "/pools/apollon/ummisco/home/fcastanet/paper_writing/Bbox_approaches/saves_models/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda"))
predictor.model.load_state_dict(torch.load("/pools/apollon/ummisco/home/fcastanet/paper_writing/Bbox_approaches/saves_models/BBS2_1024_2_epoch5.torch"))
model_yolo_1024 = YOLOv10("/pools/apollon/ummisco/home/fcastanet/objectDetection/YoloV7/yolov7/yolov10/runs/detect/train3/weights/best.pt")
image_folder_test = "/home/fcastanet/data/dataset_org"
filenames = os.listdir(image_folder_test)

for filename in filenames:
    image_path = os.path.join(image_folder_test, filename)
    image = cv2.imread(image_path)
    image_patches = patchify_with_border_handling(image, img_patch_size, step=step)
    num_patches_y, num_patches_x = image_patches.shape[:2]
    patch_height, patch_width = image_patches.shape[3:5]
    reconstructed_image_tuned = np.zeros((num_patches_y * patch_height, num_patches_x * patch_width), dtype=np.uint8)
    
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            current_image = Image.fromarray(image_patches[i, j, 0])
            results = model_yolo_1024(source=current_image, conf=0.25, save=False)[0].boxes.xyxyn.tolist()
            final_mask = np.zeros((size, size), dtype=np.uint8)
            non_contained_boxes = []
            
            for box in results:
                box = [int(elem * size) for elem in box]
                if not any(is_contained_within(box, existing_box) for existing_box in non_contained_boxes):
                    non_contained_boxes.append(box)
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            predictor.set_image(current_image)
                            masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=np.array(box)[None, :], multimask_output=False)
                            prediction = masks[np.argmax(scores)].astype(np.uint8)
                    final_mask = np.maximum(final_mask, cv2.resize(prediction, (size, size)))
            
            y_start, y_end = i * patch_height, (i + 1) * patch_height
            x_start, x_end = j * patch_width, (j + 1) * patch_width
            reconstructed_image_tuned[y_start:y_end, x_start:x_end] = final_mask
    
    H, W, _ = image.shape
    reconstructed_image_tuned = np.stack((reconstructed_image_tuned[:H, :W],) * 3, axis=-1)
    reconstruced_mask_tuned = np.where(reconstructed_image_tuned != 0, image, 0)
    cv2.imwrite(f"/home/fcastanet/data/heatmaps/{filename}", reconstruced_mask_tuned)
