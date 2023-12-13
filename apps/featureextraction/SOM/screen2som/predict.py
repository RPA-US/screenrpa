import numpy as np
import torch
import json
from .hierarchy_constructor import labels_to_soms
from .utils import *

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import copy

ELEMENTS_MODEL = "checkpoints/screen2som/elements.pt"
TEXT_MODEL = "checkpoints/screen2som/text.pt"
CONTAINER_MODEL = "checkpoints/screen2som/container.pt"
APPLEVEL_MODEL = "checkpoints/screen2som/applevel.pt"
TOP_MODEL = "checkpoints/screen2som/toplevel.pt"



def predict(image_path):
    image_pil = cv2.imread(image_path)

    detections = dict()

    # Elements level preditions
    elements_shapes = sahi_predictions(ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0)
    detections["shapes"] = elements_shapes
    detections["imageWidth"] = image_pil.shape[1]
    detections["imageHeight"] = image_pil.shape[0]

    # Text level predictions
    text_shapes = sahi_predictions(TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections["shapes"]))
    detections["shapes"].extend(text_shapes)

    # Container Level predictions
    container_shapes = yolo_prediction(CONTAINER_MODEL, image_pil, "bbox", len(detections["shapes"]))
    detections["shapes"].extend(container_shapes)

    # Application level predictions
    applevel_shapes = yolo_prediction(APPLEVEL_MODEL, image_pil, "seg", len(detections["shapes"]))
    detections["shapes"].extend(applevel_shapes)

    # Top level predictions
    toplevel_shapes = yolo_prediction(TOP_MODEL, image_pil, "seg", len(detections["shapes"]))
    detections["shapes"].extend(toplevel_shapes)

    som = labels_to_soms(copy.deepcopy(detections))

    return som

   
def yolo_prediction(model_path, image_pil, type, id_start):
    model = YOLO(model_path)

    result = json.loads(model(image_pil, conf=0.4)[0].tojson())
    shapes = json_inference_to_labelme(
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
    shapes = coco_to_labelme(
        anns, type=type, id_start=id_start
    )

    # Unload model from memory
    del detection_model
    torch.cuda.empty_cache() 

    return shapes