import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import os
from patchify import patchify
import matplotlib.pyplot as plt
import json
from PIL import Image
from ultralytics import YOLOv10
import cv2
import matplotlib.pyplot as plt
import statistics
import re

def calculate_iou(mask1, mask2):

    if mask1.shape != mask2.shape:
        raise ValueError("Les deux masques doivent avoir la même forme")
    
    # Calculer l'intersection et l'union des deux masques
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Éviter la division par zéro
    if union == 0:
        return 0.0
    
    # Calculer l'IoU
    iou = intersection / union
    return iou

def is_contained_within(box1, box2):
    """ Check if box1 is completely inside box2 """
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data




def patchify_with_border_handling(image, img_patch_size, step):
    if len(image.shape) == 2:
        H, W = image.shape

        # Calculer le nombre de patches nécessaires
        num_patches_h = int((H) / step) + 1
        num_patches_w = int((W) / step) + 1

        # Ajouter du padding pour que les dimensions soient des multiples de step
        pad_h = (num_patches_h * step) - H
        pad_w = (num_patches_w * step) - W
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')

        # Patchify l'image paddée
        image_patches = patchify(image_padded, img_patch_size, step=step)

        return image_patches
    else:
        H, W, _ = image.shape

        # Calculer le nombre de patches nécessaires
        num_patches_h = int((H) / step) + 1
        num_patches_w = int((W) / step) + 1

        # Ajouter du padding pour que les dimensions soient des multiples de step
        pad_h = (num_patches_h * step) - H
        pad_w = (num_patches_w * step) - W
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

        # Patchify l'image paddée
        image_patches = patchify(image_padded, img_patch_size, step=step)

        return image_patches
    

size = 1024
img_patch_size = (size,size,3)
mask_patch_size = (size,size)
step = size



version = "SAM2"
device = "cuda"

if version == "SAM2" :

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "./models/sam2.1_hiera_large.pt"
    model_cfg = "/Users/floriancastanet/Library/CloudStorage/OneDrive-ERGONOVACONSEIL/Documents/ProjetUMMISCO/SAM2YOLOV10/gits/sam2/sam2/sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint)) # load net
    predictor.model.load_state_dict(torch.load("models/BBS2_1024_2_epoch5.torch"))


kernel_size = 0  # faire varier la taille
kernel = np.ones((kernel_size, kernel_size), np.uint8)



model_yolo_1024 = YOLOv10("/Users/floriancastanet/Library/CloudStorage/OneDrive-ERGONOVACONSEIL/Documents/ProjetUMMISCO/SAM2YOLOV10/models/trainedyolov10.pt")


image_folder_test = f"/home/fcastanet/data/dataset_org"

list_segmented = image_folder_test




filenames = os.listdir(image_folder_test)


all_percentages = []

for filename in filenames:
    if filename not in list_segmented :
   
        boxes_to_draw = []
        box_areas = []
        mask_areas = []

        image_path = os.path.join(image_folder_test, filename)


        image = cv2.imread(image_path)

            
        img_patch_size = (size,size,3)
        step = size
            
        
        image_patches = patchify_with_border_handling(image, img_patch_size, step=step)

        num_patches_y, num_patches_x = image_patches.shape[:2]
        patch_height, patch_width = image_patches.shape[3:5]

        reconstructed_image_tuned = np.zeros((num_patches_y * patch_height, num_patches_x * patch_width), dtype=np.uint8)
        
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                current_image = image_patches[i, j, 0]
                current_image = Image.fromarray(current_image)

                total_bbox_area = 0 

                # Getting bounding box coordinates

                results = model_yolo_1024(source=current_image, conf=0.25, save=False)
                results = results[0].boxes.xyxyn
                results = results.tolist()

                final_mask = np.zeros((size, size), dtype=np.uint8)

                if results:
                    non_contained_boxes = []
                    for box in results:
                        box = [int(elem * size) for elem in box]
                        is_contained = False
                        for existing_box in non_contained_boxes:
                            if is_contained_within(box, existing_box):
                                is_contained = True
                                break
                        non_contained_boxes.append(box)
             
                        if version == "SAM2":
                            box = np.array(box)
                            with torch.no_grad():
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    predictor.set_image(current_image)
                                    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=False)                         

                                    index_best_score = np.argsort(scores)[-1]
                                    prediction = masks[index_best_score].astype(np.uint8)


                        prediction = cv2.resize(prediction, (size, size))
                        #masque_dilate = cv2.dilate(prediction, kernel, iterations=1)
                        final_mask = np.maximum(final_mask, prediction)

                        y_start, y_end = i * patch_height, (i + 1) * patch_height
                        x_start, x_end = j * patch_width, (j + 1) * patch_width
                        reconstructed_image_tuned[y_start:y_end, x_start:x_end] = final_mask

                        start_point = (x_start + box[0], y_start + box[1])
                        end_point = (x_start + box[2], y_start + box[3])

                        if not is_contained:
                            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
                            total_bbox_area += bbox_area

                        boxes_to_draw.append((start_point, end_point))




                box_areas.append(total_bbox_area)
                mask_areas.append(np.sum(final_mask[final_mask != 0]))


        H, W, C = image.shape
        reconstructed_image_tuned = reconstructed_image_tuned[:H, :W]
        reconstructed_image_tuned = np.stack((reconstructed_image_tuned, reconstructed_image_tuned, reconstructed_image_tuned), axis=-1)
        reconstruced_mask_tuned = np.zeros_like(image)
        reconstruced_mask_tuned[reconstructed_image_tuned != 0] = image[reconstructed_image_tuned != 0]

        color = (0, 255, 0)
        thickness = 10

        reconstructed_mask_area = np.sum(reconstructed_image_tuned != 0)

        print("saving img")
        cv2.imwrite(f"/home/fcastanet/data/heatmaps/{filename}", reconstruced_mask_tuned)
