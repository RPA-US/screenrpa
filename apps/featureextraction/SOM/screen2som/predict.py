import numpy as np
import torch
import json
import cv2
from .hierarchy_constructor import labels_to_output
from .utils import *
from shapely.geometry import Polygon

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import copy

ELEMENTS_MODEL = "checkpoints/screen2som/elements.pt"
TEXT_MODEL = "checkpoints/screen2som/text.pt"
CONTAINER_MODEL = "checkpoints/screen2som/container.pt"
APPLEVEL_MODEL = "checkpoints/screen2som/applevel.pt"
TOP_MODEL = "checkpoints/screen2som/toplevel.pt"



def predict(image_path, path_to_save_bordered_images):
    image_pil = cv2.imread(image_path)

    detections = dict()
    detections["img_shape"] = image_pil.shape

    image_pil  = cv2.resize(image_pil, (640, 360))

    # Elements level preditions
    elements_compos = sahi_predictions(ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0)
    detections["compos"] = elements_compos
    detections["img_shape"] = image_pil.shape

    # Text level predictions
    text_compos = sahi_predictions(TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections["compos"]) + 1)
    detections["compos"].extend(text_compos)

    # Container Level predictions
    container_compos = yolo_prediction(CONTAINER_MODEL, image_pil, "bbox", len(detections["compos"]) + 1)
    detections["compos"].extend(container_compos)

    # Application level predictions
    applevel_compos = yolo_prediction(APPLEVEL_MODEL, image_pil, "seg", len(detections["compos"]) + 1)
    detections["compos"].extend(applevel_compos)

    # Top level predictions
    toplevel_compos = yolo_prediction(TOP_MODEL, image_pil, "seg", len(detections["compos"]) + 1)
    detections["compos"].extend(toplevel_compos)

    # Order compos by area
    detections["compos"].sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    # Resize detections
    for compo in detections["compos"]:
        for point in compo["points"]:
            # Update the point to match new dimensions
            point[0] = point[0] * detections["img_shape"][1] / image_pil.shape[1]
            point[1] = point[1] * detections["img_shape"][0] / image_pil.shape[0]

    # Image crops from shapes
    recortes = []

    for i, shape in enumerate(detections["compos"]):
        x1, y1 = np.min(shape["points"], axis=0)
        x2, y2 = np.max(shape["points"], axis=0)
        recortes.append(image_pil[int(y1):int(y2), int(x1):int(x2)])

    # SOM from shapes
    predictions = labels_to_output(copy.deepcopy(detections))

    # Save bordered images
    save_bordered_images(image_path, detections["compos"], path_to_save_bordered_images)

    return recortes, predictions

   
def yolo_prediction(model_path, image_pil, type, id_start):
    model = YOLO(model_path)

    result = json.loads(model(image_pil, conf=0.4)[0].tojson())
    shapes = json_inference_to_compos(
        result, type=type, id_start=id_start
    )

    # Unload model from memory
    del model
    torch.cuda.empty_cache()

    return shapes

def sahi_predictions(model_path, image_pil, slice_width, slice_height, overlap, type, id_start):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.4,
    )

    result = get_sliced_prediction(
        image_pil,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
    )
    anns = result.to_coco_annotations()
    shapes = coco_to_compos(
        anns, type=type, id_start=id_start
    )

    # Unload model from memory
    del detection_model
    torch.cuda.empty_cache() 

    return shapes