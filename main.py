import math
import cv2
import numpy as np
import pandas as pd
import os


def main():
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
    # Thanks for this tutorial kind stranger
    # https://answers.opencv.org/question/136796/turning-aruco-marker-in-parallel-with-camera-plane/
    # ToDo postponed

    # 3. pink mask
    pink_mask_before = create_pink_mask(img_before)
    pink_mask_after = create_pink_mask(img_after)

    # 4. detect (pink) circle
    circle_mask_before = detect_and_mask_circle(pink_mask_before)
    circle_mask_after = detect_and_mask_circle(pink_mask_after)

    # 5. bitmask (only detect inside masked region)
    petri_dish_before = cv2.bitwise_and(img_before, img_before, mask=circle_mask_before)
    petri_dish_after = cv2.bitwise_and(img_after, img_after, mask=circle_mask_after)

    # 6. detect yellow
    blob_mask_before = create_yellow_mask(petri_dish_before)
    blob_mask_after = create_yellow_mask(petri_dish_after)

    # 7. calculate (bit) area
    before_pixel_area = cv2.countNonZero(blob_mask_before)
    after_pixel_area = cv2.countNonZero(blob_mask_after)

    # 8. calculate aruco area for comparison (from bullet point 2)
    # https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    # Aruco detect, Marker is 2.70 cm x 2.70 cm
    ratio_before = calculate_pixel_to_cm_ratio(img_before, 2.7)
    ratio_after = calculate_pixel_to_cm_ratio(img_before, 2.7)

    # 9. Do math and calculate to scale
    # Todo
    # calculate_area_cm2(before_pixel_area, ratio_before
    # thank u kind stranger
    # https://stackoverflow.com/questions/64394768/how-calculate-the-area-of-irregular-object-in-an-image-opencv
    # 10. Safe results in cv
    # Todo
    # 11. Do for every image pair
    # Todo
    # 12. Success - hopefully
    # Todo
    return


def create_pink_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])  # Default
    upper_pink = np.array([170, 255, 255])  # Default

    return finetune_mask('pink for circle detection', hsv, img, lower_pink, upper_pink)


def finetune_mask(windowname, hsv_img, original_image, lower, upper):
    while 1:
        mask = cv2.inRange(hsv_img, lower, upper)
        debug_imshow(windowname, cv2.bitwise_and(original_image, original_image, mask=mask))
        debug_imshow('original', original_image)
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


def detect_and_mask_circle(img):
    # Convert to grayscale.
    gray = img

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=80, maxRadius=4000)

    mask = np.zeros_like(img)
    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        pt = detected_circles[0, 0]
        a, b, r = pt[0], pt[1], pt[2]
        # Draw the circumference of the circle.
        cv2.circle(mask, (a, b), r, (255, 255, 255), -1)

    return mask

def create_yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])  # Adjust these values based on the image
    upper_yellow = np.array([30, 255, 255])  # Adjust these values based on the image

    return finetune_mask('yellow for blob detection', hsv, img, lower_yellow, upper_yellow)

def calculate_pixel_to_cm_ratio(img, aruco_side_length):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(img)
    cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

    aruco_perimeter_sum = 0
    for c in markerCorners:
        aruco_perimeter = cv2.arcLength(markerCorners[0], True) #First 1, then all 4
        aruco_perimeter_sum += aruco_perimeter

    mean_aruco_perimeter = aruco_perimeter_sum / len(markerCorners)
    pixel_cm_ratio = mean_aruco_perimeter / (aruco_side_length * 4)

    return pixel_cm_ratio

def debug_imshow(window_name, img):
    #For debugging
    ratio = 0.2
    size = (math.floor(3000 * ratio), math.floor(4000 * ratio))
    cv2.imshow('debug: ' + window_name, cv2.resize(img, size))


if __name__ == "__main__":
    main()
