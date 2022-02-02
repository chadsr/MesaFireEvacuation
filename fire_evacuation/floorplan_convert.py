import cv2
import sys
import numpy as np

sys.path.append("../")

from fire_evacuation.symbols_to_obstacles import add_obstacles_to_GAN
from fire_evacuation.image_boundary import get_final_mask

RED_LOWER_THRES = [(0, 50, 50), (20, 255, 255)]
RED_UPPER_THRES = [(150, 50, 50), (180, 255, 255)]
BLUE_THRES = [(80, 50, 50), (150, 255, 255)]
GREEN_THRES = [(35, 50, 50), (85, 255, 255)]
BLACK_THRES = [(0, 0, 0), (0, 0, 0)]

def add_border_img(img):
    bordersize = 10
    row, col = img.shape[:2]
    bottom = img[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    
    return border


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
combined_img = add_border_img(combined_img)
cv2.imwrite("../input/images/downsample.png", combined_img)


txt_floorplan = np.zeros(combined_img.shape[:2], 'U1')
txt_floorplan.fill('E')

walls = get_wall_image_layer(combined_img)
doors = get_door_image_layer(combined_img)
obstacles = get_obstacle_image_layer(combined_img)

final_mask = get_final_mask(combined_img, "GAN_label.png" )

txt_floorplan[np.where(final_mask == 0)[:2]] = "_"
txt_floorplan[np.where(walls != (255, 255, 255))[:2]] = "W"
txt_floorplan[np.where(doors != (255, 255, 255))[:2]] = "D"
txt_floorplan[np.where(obstacles != (255, 255, 255))[:2]] = "F"

np.savetxt(f"floorplans/test_floorplan.txt", txt_floorplan, fmt="%s")
