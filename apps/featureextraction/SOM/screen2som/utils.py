import numpy as np
import cv2
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


def save_bordered_images(img_path, detected_shapes, path_to_save_bordered_images):
    # Give a distinct different color to each label
    tint_colors = {
        "Application": (32, 200, 255),
        "Dock": (100, 33, 30),
        "Taskbar": (210, 45, 215),
        "Header": (55, 155, 255),
        "Scrollbar": (0, 123, 50),
        "Toolbar": (255, 111, 94),
        "BrowserToolbar": (180, 0, 200),
        "TabActive": (75, 0, 130),
        "TabInactive": (238, 130, 238),
        "Sidebar": (255, 0, 0),
        "Navbar": (0, 255, 0),
        "Container": (0, 0, 255),
        "Image": (255, 255, 0),
        "BrowserURLInput": (0, 255, 255),
        "WebIcon": (255, 0, 255),
        "Icon": (192, 192, 192),
        "Switch": (128, 128, 128),
        "BtnSq": (128, 0, 0),
        "BtnPill": (128, 128, 0),
        "BtnCirc": (0, 128, 0),
        "CheckboxChecked": (128, 0, 128),
        "CheckboxUnchecked": (0, 128, 128),
        "RadiobtnSelected": (0, 0, 128),
        "RadiobtnUnselected": (255, 165, 0),
        "TextInput": (255, 20, 147),
        "Dropdown": (220, 20, 60),
        "Link": (50, 205, 50),
        "Text": (70, 130, 180),
    }

    img = cv2.imread(img_path)
    for i in range(len(detected_shapes)):
        # Draw Polygons 
        cv2.polylines(
            img,
            np.int32([detected_shapes[i]["points"]]),
            True,
            tint_colors[detected_shapes[i]["class"]],
            2,
        )

        img_aux = img.copy()
        cv2.fillPoly(
            img_aux,
            np.int32([detected_shapes[i]["points"]]),
            tint_colors[detected_shapes[i]["class"]],
        )

        img = cv2.addWeighted(img_aux, 0.2, img, 0.8, 0)

    cv2.imwrite(path_to_save_bordered_images + img_path.split("/")[-1] + "_bordered.png", img)

def coco_to_compos(coco_anns, type="bbox", id_start=1):
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
                    "class": ann["category_name"],
                    "text": "", #TODO,
                    "points": points,
                    "centroid": list(Polygon(points).centroid.coords[0])
                }
            )

        elif type == "seg":
            points = np.array(ann["segmentation"][0]).reshape(-1, 2)

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "class": ann["category_name"],
                    "text": "", #TODO,
                    "points": points.tolist(),
                    "centroid": list(Polygon(points).centroid.coords[0])
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")

        for i, shape in enumerate(res):
            shape["id"] = i + id_start
    
    return res



def json_inference_to_compos(anns, type="bbox", id_start=1):
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
                    "class": ann["name"],
                    "text": "", #TODO,
                    "points": points,
                    "centroid": list(Polygon(points).centroid.coords[0])
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
                    "class": ann["name"],
                    "text": "", #TODO,
                    "points": points.tolist(),
                    "centroid": list(Polygon(points).centroid.coords[0])
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")
        
        for i, shape in enumerate(res):
            shape["id"] = i + id_start

    return res