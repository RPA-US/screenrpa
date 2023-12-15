from shapely.geometry import Polygon

def build_tree(tree: list, depth=1):
    """
    Recursively constructs a tree hierarchy from a list of shapes.

    Args:
        tree (list): A list of shapes
        depth (int): The current depth of the tree.

    Returns:
        list: A tree representing the hierarchy of the shapes.
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
            except ZeroDivisionError:
                continue
        if len(shape1["children"]) > 0:
            shape1["children"] = build_tree(shape1["children"], depth=depth + 1)
    return list(filter(lambda s: s["depth"] == depth, tree))

def ensure_toplevel(tree:dict, bring_up=None):
    """
    Ensures that the TopLevel labels are in the top level of the tree

    Args:
        tree (list): A tree representing the hierarchy of the shapes.

    Returns:
        tree: A tree representing the hierarchy of the shapes.
    """
    if bring_up is None:
        bring_up = []
    children = tree["children"]
    for child in children:
        if child["type"] == "node":
            child["children"], bring_up = ensure_toplevel(child, bring_up=bring_up)
            if len(child["children"]) == 0:
                child["type"] = "leaf"
            if child["label"] in ["Application", "Taskbar", "Dock"]:
                bring_up.append(child)
            
    new_children = list(filter(lambda c: c not in bring_up, children))

    if tree["type"] == "root":
        new_children.extend(bring_up)
        new_children = readjust_depth(new_children, 1)

    return new_children, bring_up

def readjust_depth(nodes, depth):
    for node in nodes:
        node["depth"] = depth
        node["children"] = readjust_depth(node["children"], depth+1)

    return nodes

def labels_to_soms(labels):
    """
    Converts a list of labels into  a SOM .

    Args:
        dataset_labels (list): list of predictions.

    Returns:
        dict: SOM.
    """
    shapes = labels["shapes"]
    for shape in shapes:
        shape["depth"] = 1
        shape["type"] = "leaf"

    shapes.sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    som = {
        "depth": 0,
        "type": "root",
        "points": [
            [0, 0],
            [0, labels["imageHeight"]],
            [labels["imageWidth"], 0],
            [labels["imageWidth"], labels["imageHeight"]],
        ],
        "children": build_tree(shapes),
    }

    som["children"], _ = ensure_toplevel(som)
    
    return som