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

from core.settings import SCREEN2SOM_CONFIG_PATH

def predict(image_path, img_index, path_to_save_bordered_images, text_detected_by_OCR):
    # Set screen2som models
    json_config = json.load(open(SCREEN2SOM_CONFIG_PATH, "r"))

    resize_img_size = json_config["img_size"]
    models = json_config["models"]

    image_pil = cv2.imread(image_path)

    detections = dict()
    detections["img_shape"] = image_pil.shape
    detections["compos"] = []

    image_pil_resized  = cv2.resize(image_pil, resize_img_size)

    for model_name, properties in models.items():
        id_start = 0 if len(detections["compos"]) == 0 else len(detections["compos"]) + 1
        if properties["function"] == "yolo":
            compos = yolo_prediction(os.path.join(json_config["models_path"], model_name + ".pt"), image_pil_resized, properties["type"], id_start)
            detections["compos"].extend(compos)
        else:
            sahi_x = json_config["sahi_size"][0]
            sahi_y = json_config["sahi_size"][1]
            compos = sahi_predictions(os.path.join(json_config["models_path"], model_name + ".pt"), image_pil_resized, sahi_x, sahi_y, 0.3, properties["type"], id_start)
            detections["compos"].extend(compos)

    # # Elements level preditions
    # elements_compos = sahi_predictions(ELEMENTS_MODEL, image_pil_resized, 640, 360, 0.3, "bbox", 0)
    # detections["compos"] = elements_compos

    # # Text level predictions
    # text_compos = sahi_predictions(TEXT_MODEL, image_pil_resized, 640, 360, 0.3, "bbox", len(detections["compos"]) + 1)
    # detections["compos"].extend(text_compos)

    # # Container Level predictions
    # container_compos = yolo_prediction(CONTAINER_MODEL, image_pil_resized, "seg", len(detections["compos"]) + 1)
    # detections["compos"].extend(container_compos)

    # # Application level predictions
    # applevel_compos = yolo_prediction(APPLEVEL_MODEL, image_pil_resized, "bbox", len(detections["compos"]) + 1)
    # detections["compos"].extend(applevel_compos)

    # # Top level predictions
    # toplevel_compos = yolo_prediction(TOP_MODEL, image_pil_resized, "seg", len(detections["compos"]) + 1)
    # detections["compos"].extend(toplevel_compos)

    # Order compos by area
    detections["compos"].sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    # Resize detections
    for compo in detections["compos"]:
        for point in compo["points"]:
            # Update the point to match original dimensions
            point[0] = point[0] * detections["img_shape"][1] / image_pil_resized.shape[1] 
            point[1] = point[1] * detections["img_shape"][0] / image_pil_resized.shape[0]
        compo["centroid"] = list(Polygon(compo["points"]).centroid.coords[0])

    # Image crops from shapes
    recortes = []

    for i, shape in enumerate(detections["compos"]):
        x1, y1 = np.min(shape["points"], axis=0)
        x2, y2 = np.max(shape["points"], axis=0)
        recortes.append(image_pil[int(y1):int(y2), int(x1):int(x2)])

    merge_text_with_OCR(detections, img_index, text_detected_by_OCR)

    # SOM from shapes
    predictions = labels_to_output(copy.deepcopy(detections))

    # Save bordered images
    save_bordered_images(image_path, detections["compos"], path_to_save_bordered_images, json_config["classes"])

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