import numpy as np
from shapely.geometry import Polygon


def detect_duplicates(detected_shapes):
    # Create a 2d numpy array of shape (len(detected_shapes), len(detected_shapes))
    duplicates = []
    for i in range(len(detected_shapes)):
        for j in range(i + 1, len(detected_shapes)):
            if i == j:
                continue
            detected_polygon1 = Polygon(detected_shapes[i]["points"])
            detected_polygon2 = Polygon(detected_shapes[j]["points"])
            intersection = detected_polygon1.intersection(detected_polygon2).area
            union = detected_polygon1.union(detected_polygon2).area
            if intersection / union >= 0.5:
                duplicates.append((detected_polygon1, detected_polygon2))

    return duplicates


def coco_to_labelme(coco_anns, type="bbox", id_start=0):
    res = []
    for i, ann in enumerate(coco_anns):
        if type == "bbox":
            x, y, w, h = ann["bbox"]
            points = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["category_name"],
                    "points": points,
                }
            )

        elif type == "seg":
            points = np.array(ann["segmentation"][0]).reshape(-1, 2)

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["category_name"],
                    "points": points.tolist(),
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")

        for i, shape in enumerate(res):
            shape["id"] = i + id_start
    
    return res



def json_inference_to_labelme(anns, type="bbox", id_start=0):
    res = []
    for i, ann in enumerate(anns):
        if type == "bbox":
            box = ann["box"]

            points = [
                [box["x1"], box["y1"]],
                [box["x2"], box["y1"]],
                [box["x2"], box["y2"]],
                [box["x1"], box["y2"]],
            ]

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["name"],
                    "points": points,
                }
            )

        elif type == "seg":
            x_points = ann["segments"]["x"]
            y_points = ann["segments"]["y"]
            points = np.array([x_points, y_points]).T

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["name"],
                    "points": points.tolist(),
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")
        
        for i, shape in enumerate(res):
            shape["id"] = i + id_start

    return res