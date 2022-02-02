import cv2
import numpy as np
from fire_evacuation.symbols_to_obstacles import add_obstacles_to_GAN


RED_LOWER_THRES = [(0, 50, 50), (20, 255, 255)]
RED_UPPER_THRES = [(150, 50, 50), (180, 255, 255)]
BLUE_THRES = [(80, 50, 50), (150, 255, 255)]
GREEN_THRES = [(35, 50, 50), (85, 255, 255)]
BLACK_THRES = [(0, 0, 0), (0, 0, 0)]


def color_threshold(img, thresholds=[]):
    result = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = None
    for thres in thresholds:
        # Blue color
        lower = np.array(thres[0])
        upper = np.array(thres[1])
        mask = cv2.inRange(img, lower, upper)
        if masks is None:
            masks = mask
        else:
            masks = masks + mask

    masks = cv2.medianBlur(masks, 3)
    result[masks != 255] = 255

    return result


def get_wall_image_layer(img):
    return color_threshold(
        img,
        [BLUE_THRES],
    )


def get_window_image_layer(img):
    return color_threshold(
        img,
        [GREEN_THRES],
    )


def get_wall_window_image_layer(img):
    return color_threshold(
        img,
        [BLUE_THRES, GREEN_THRES],
    )


def get_door_image_layer(img):
    return color_threshold(
        img,
        [
            RED_LOWER_THRES,
            RED_UPPER_THRES,
        ],
    )


def get_obstacle_image_layer(img):
    return color_threshold(
        img,
        [BLACK_THRES]
    )

combined_img = add_obstacles_to_GAN("GAN_label.png")
combined_img = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(combined_img)))
cv2.imwrite("../input/images/downsample.png", combined_img)


txt_floorplan = np.zeros(combined_img.shape[:2], 'U1')
txt_floorplan.fill('_')

walls = get_wall_image_layer(combined_img)
doors = get_door_image_layer(combined_img)
obstacles = get_obstacle_image_layer(combined_img)

txt_floorplan[np.where(walls != (255, 255, 255))[:2]] = "W"
txt_floorplan[np.where(doors != (255, 255, 255))[:2]] = "E"
txt_floorplan[np.where(obstacles != (255, 255, 255))[:2]] = "F"

np.savetxt(f"floorplans/test_floorplan.txt", txt_floorplan, fmt="%s")
