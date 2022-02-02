import cv2
import numpy as np

# list of non-navigable obstacles in a floorplan
OBSTACLES = ["bath", "sink", "double sink", "toilet", "desk1", "row_desk_attched_wall", "rectangular-sink",
             "round-table_with_seat", "washroom-sink", "room-sink", "chair", "couch", "bed"]


def get_classes():
    class_id = 0
    id_to_class = {}
    with open(f"../input/labels/classes.txt") as f:
        classes = f.readlines()
        for c in classes:
            id_to_class[class_id] = c
    return id_to_class


def unconvert(class_id, width, height, x, y, w, h):
    """
    Converts the normalized positions  into integer positions
    """
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


def get_symbol_coords(height, width, symbol_labels):
    labels_list = []
    with open(f"../input/labels/{symbol_labels}") as f:
        labels = f.readlines()
        for l in labels:
            label_comps = l.split(" ")
            labels_list.append(unconvert(label_comps[0], width, height, label_comps[1],
                                         label_comps[2], label_comps[3], label_comps[4]))
    return labels_list


def get_obstacle_img(fp_filename):
    fp_img = cv2.imread(f"../input/images/{fp_filename}")
    height, width = fp_img.shape[:2]

    img = np.zeros([height, width, 3], dtype=np.uint8)
    img[:] = 255

    objects = get_symbol_coords(height, width, "mappedin_YOLO60.txt")

    id_to_class = get_classes()
    for obj in objects:
        symbol_class = id_to_class[int(obj[0])]
        # only draw the obstacle if it is in the OBSTACLES list
        if symbol_class in OBSTACLES:
            start = (obj[1], obj[3])
            end = (obj[2], obj[4])
            cv2.rectangle(img, start, end, (0, 0, 0), -1)
    cv2.imwrite("../input/images/test_img.png", img)
    return img
