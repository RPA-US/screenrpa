from shapely.geometry import Polygon
import copy

def build_tree(tree: list, depth=1, text_class="Text"):
    """
    Recursively constructs a tree hierarchy from a list of compos.

    Args:
        tree (list): A list of compos
        depth (int): The current depth of the tree.

    Returns:
        list: A tree representing the hierarchy of the compos.
    """
    for shape1 in tree:
        if shape1["depth"] != depth:
            continue
        shape1["children"] = []
        for shape2 in tree:
            if shape2["depth"] != depth or shape1 == shape2:
                continue
            polygon1 = Polygon(shape1["points"])
            polygon2 = Polygon(shape2["points"])
            if polygon1.is_valid == False or polygon2.is_valid == False:
                continue
            intersection = polygon1.intersection(polygon2).area
            try:
                if intersection / polygon2.area > 0.5:
                    shape2["depth"] = depth + 1
                    shape1["children"].append(shape2)
                    shape1["type"] = "node"
                    shape2["xpath"].append(shape1["id"])
                    if shape2["class"] == text_class:
                        shape1["text"] += shape2["text"] + " | "
            except ZeroDivisionError:
                continue
        if len(shape1["children"]) > 0:
            shape1["children"] = build_tree(shape1["children"], depth=depth + 1)
    return list(filter(lambda s: s["depth"] == depth, tree))

def ensure_toplevel(tree:dict, bring_up=None):
    """
    Ensures that the TopLevel labels are in the top level of the tree

    Args:
        tree (list): A tree representing the hierarchy of the compos.

    Returns:
        tree: A tree representing the hierarchy of the compos.
    """
    if bring_up is None:
        bring_up = []
    children = tree["children"]
    for child in children:
        if child["type"] == "node":
            child["children"], bring_up = ensure_toplevel(child, bring_up=bring_up)
            if len(child["children"]) == 0:
                child["type"] = "leaf"
            if child["class"] in ["Application", "Taskbar", "Dock"]:
                bring_up.append(child)
                # remove 1st elements from list(stack), if any
                if len(child["xpath"]) > 0:
                    child["xpath"].pop(0)
            
    new_children = list(filter(lambda c: c not in bring_up, children))

    if tree["type"] == "root":
        new_children.extend(bring_up)
        new_children = readjust_depth(new_children, 1)

    return new_children, bring_up

def readjust_depth(nodes, depth):
    for node in nodes:
        # Remove xpath elements no longer needed
        node["xpath"] = node["xpath"][node["depth"]-depth:]
        # Readjust depth
        node["depth"] = depth
        node["children"] = readjust_depth(node["children"], depth+1)

    return nodes

def labels_to_output(labels, text_class="Text"):
    """
    Converts a list of labels into  a SOM .

    Args:
        dataset_labels (list): list of predictions.

    Returns:
        dict: SOM.
    """
    compos = labels["compos"]
    for shape in compos:
        shape["depth"] = 1
        shape["type"] = "leaf"
        shape["xpath"] = []

    compos.sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    som = {
        "depth": 0,
        "type": "root",
        "id": 0,
        "points": [
            [0, 0],
            [0, labels["img_shape"][0]],
            [labels["img_shape"][1], 0],
            [labels["img_shape"][1], labels["img_shape"][0]]
        ],
        "centroid": list(Polygon([
            [0, 0],
            [0, labels["img_shape"][0]],
            [labels["img_shape"][1], 0],
            [labels["img_shape"][1], labels["img_shape"][0]]
        ]).centroid.coords[0]),
        "children": build_tree(copy.deepcopy(compos), text_class=text_class),
    }


    som["children"], _ = ensure_toplevel(som)

    # Copy XPath of som compos to compos
    for compo in compos:
        for node in flatten_som(som["children"]):
            if compo["id"] == node["id"]:
                compo["xpath"] = node["xpath"]
                compo["depth"] = node["depth"]
                compo["type"] = node["type"]
                break

    labels["som"] = som
    
    return labels

def flatten_som(tree):
    """
    Flatten the SOM tree into a list of nodes.

    Args:
        tree (list): A tree representing the hierarchy of the SOM.

    Returns:
        list: A flattened list of nodes.
    """
    flattened = []
    for node in tree:
        flattened.append(node)
        if node["type"] == "node":
            flattened.extend(flatten_som(node["children"]))
    return flattened