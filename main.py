import math
import cv2
import numpy as np
import pandas as pd
import os


def main():
    # What to do

    # 1. Read in image
    data_folder = 'data'
    experiment_folders = os.listdir(data_folder)  # Get Experiment Names
    number_files = len(experiment_folders)  # Get number of experiments

    results_folder = 'results'
    for folder in experiment_folders:
        # Versuchsdurchlauf 1 bc of science yeah
        output_folder = os.path.join(results_folder, folder, 'Versuchsdurchlauf 1')
        os.makedirs(output_folder, exist_ok=True)

    # From now on every temp file can be stored the fitting result folder for better analyzes

    # Load images
    # ToDo: Multiple images. For now just the Test images in data/1
    input_folder = os.path.join(data_folder, '1')
    img_before = cv2.imread(os.path.join(input_folder, 'before.jpg'))
    img_after = cv2.imread(os.path.join(input_folder, 'after.jpg'))

    # 2. perspective removal, correction
    # ToDo postponed

    # 3. pink mask
    pink_mask_before = create_pink_mask(img_before)
    pink_mask_after = create_pink_mask(img_after)

    # 4. detect (pink) circle
    # 5. bitmask (only detect inside masked region)
    # 6. detect yellow
    # 7. calculate (bit) area
    # 8. calculate aruco area for comparison (from bullet point 2)
    # 9. Do math and calculate to scale
    # 10. Safe results in cv
    # 11. Do for every image pair
    # 12. Success
    return


def create_pink_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])  # Default
    upper_pink = np.array([170, 255, 255])  # Default

    return finetune_mask('pink  for circle detection', hsv, lower_pink, upper_pink)


def finetune_mask(windowname, hsv_img, lower, upper):
    while 1:
        mask = cv2.inRange(hsv_img, lower, upper)
        debug_imshow(windowname, mask)
        k = cv2.waitKey(0)
        print(k)
        if k == 27:
            print('ready')
            cv2.destroyAllWindows()
            break
        if k == 113:  # w
            lower = lower + [1, 0, 0]
            print('lower: h++')
        if k == 97:  # s
            lower = lower + [-1, 0, 0]
            print('lower: h--')
        if k == 115:  # a
            upper = upper + [-1, 0, 0]
            print('upper: h--')
        if k == 119:  # d
            upper = upper + [1, 0, 0]
            print('upper: h++')

        if k == 101:  # w
            lower = lower + [0, 1, 0]
            print('lower: s++')
        if k == 100:  # s
            lower = lower + [0, -1, 0]
            print('lower: s--')
        if k == 102:  # a
            upper = upper + [0, -1, 0]
            print('upper: s--')
        if k == 114:  # d
            upper = upper + [0, 1, 0]
            print('upper: s++')

        if k == 116:  # w
            lower = lower + [0, 0, 1]
            print('lower: v++')
        if k == 103:  # s
            lower = lower + [0, 0, -1]
            print('lower: v--')
        if k == 122:  # a
            upper = upper + [0, 0, 1]
            print('upper: v--')
        if k == 104:  # d
            upper = upper + [0, 0, -1]
            print('upper: v++')

    return mask


def debug_imshow(window_name, img):
    #For debugging
    ratio = 0.2
    size = (math.floor(3000 * ratio), math.floor(4000 * ratio))
    cv2.imshow('debug: ' + window_name, cv2.resize(img, size))


if __name__ == "__main__":
    main()
