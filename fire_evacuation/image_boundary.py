import os
from typing_extensions import final
import cv2
import numpy as np


def process_img(img):
    """
    Processes the image using thresholding and dilation to connect components of the image that are close together.
    :param img: The original image
    :return: The thresholded/dilated image and the original image in grayscale
    """
    kernel = np.ones((3, 3), np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh, gray


def find_outline(img):
    """
    Finds the contour around objects in the image
    :param img: The original image (grayscale)
    :return: The external contours around all objects in the image.
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    outlines = []
    hierarchy = hierarchy[0]

    for component in zip(contours, hierarchy):
        cnt = component[0]
        outlines.append(cnt)
    return outlines


def mask_outline(gray, comp_mask, out_conts):
    """
    Creates a mask from the outline contour to remove anything outside the largest outline in the image.
    In case that outline is very large, the connected component mask is used to reduce computation time.
    :param gray: The grayscale image
    :param comp_mask: The largest connected components as a mask
    :param out_cont: The outline contour around the largest connected components
    :return: The output from the masked outline
    """
    mask = np.zeros_like(gray)
    out = np.ones(gray.shape) * 255

    # if the outline is of a reasonable size, use it to create a mask
    if len(out_conts) < 5000:
        for cnt in out_conts:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            out[mask == 255] = gray[mask == 255]
    else:
        print("Using component mask")
        mask = comp_mask
        out[mask == 255] = gray[mask == 255]

    return out, mask


def get_largest_components(thresh):
    """
    Find the largest components in the image.
    :param thresh: The thresholded/inverted image.
    :return: The largest connected components as a mask.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]

    sizes = list(sizes)

    # get indices of 3 largest components (handles multiple floor plans in one image)
    largest_i = np.argsort(sizes)[-3:]

    max_size = sizes[largest_i[-1]]

    img2 = np.zeros((output.shape))
    # create mask from the largest components (if larger than lower limit)
    for i in largest_i:
        # if the component is at least 75% the size of the largest component, keep it
        if sizes[i]/max_size > 0.75:
            img2[output == i + 1] = 255
    img2 = img2.astype(np.uint8)
    return img2



def get_final_mask(img, img_name):
    out_dir = "../input/"
    thresh, gray = process_img(img)
    img2 = get_largest_components(thresh)
    outlines = find_outline(img2)
    
    if len(outlines) > 0:
        output, mask = mask_outline(gray, img2, outlines)
        mask = (255-mask)*output
        final_mask = gray*mask
        cv2.imwrite(out_dir + "mask_" + img_name, final_mask )
    
    return final_mask
