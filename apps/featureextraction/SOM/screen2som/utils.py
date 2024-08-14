import os
import cv2
import numpy as np
from shapely.geometry import Polygon, Point


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


def save_bordered_images(img_path, detected_shapes, path_to_save_bordered_images, tint_colors):
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

    cv2.imwrite(os.path.join(path_to_save_bordered_images, os.path.basename(img_path) + "_bordered.png"), img)

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

def merge_text_with_OCR(detections, img_index, text_detected_by_OCR):
    # Store on global_y all the "y" coordinates and text boxes
    # Each row is a different text box, much more friendly than the format returned by keras_ocr 
    global_y = []
    global_x = []
    words = {}
    words[img_index] = {}

    if len(text_detected_by_OCR) > 0:
        for j in range(0, len(text_detected_by_OCR[img_index])):
            coordenada_y = []
            coordenada_x = []

            for i in range(0, len(text_detected_by_OCR[img_index][j][1])):
                coordenada_y.append(text_detected_by_OCR[img_index][j][1][i][1])
                coordenada_x.append(text_detected_by_OCR[img_index][j][1][i][0])

            word = text_detected_by_OCR[img_index][j][0]
            centroid = (np.mean(coordenada_x), np.mean(coordenada_y))
            if word in words[img_index]:
                words[img_index][word] += [centroid]
            else:
                words[img_index][word] = [centroid]

            global_y.append(coordenada_y)
            global_x.append(coordenada_x)
            # print('Coord y, cuadro texto ' +str(j+1)+ str(global_y[j]))
            # print('Coord x, cuadro texto ' +str(j+1)+ str(global_x[j]))

    for i, compo in enumerate(detections["compos"]):
        if compo["class"] == "Text":
            for word in words[img_index]:
                for centroid in words[img_index][word]:
                    if Polygon(compo["points"]).contains(Point([centroid])):
                        # Add word to the text box
                        if compo["text"] == "":
                            compo["text"] = word
                        else:
                            compo["text"] += " " + word