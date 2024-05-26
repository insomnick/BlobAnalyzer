import math

import cv2
import numpy as np
import pandas as pd
import os

def align_image(img):
    # Thanks for this tutorial kind stranger
    # https://answers.opencv.org/question/136796/turning-aruco-marker-in-parallel-with-camera-plane/
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray1)

    if ids is not None and len(corners) > 0:
        aligned_img1, aligned_img2 = [], []
        return aligned_img1, aligned_img2
    else:
        raise ValueError("ArUco markers not detected in both images.")


def extract_color_region(img, lower_color, upper_color, circle_mask):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask for the specified color range
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    # Mask only the specified color area within the circle
    color_region = cv2.bitwise_and(color_mask, color_mask, mask=cv2.bitwise_not(circle_mask))

    return color_region


def calculate_area(mask):
    return cv2.countNonZero(mask)


def main(experiment_id):
    data_folder = 'data'
    results_folder = 'results'

    input_folder = os.path.join(data_folder, experiment_id)
    output_folder = os.path.join(results_folder, experiment_id)

    os.makedirs(output_folder, exist_ok=True)

    # Load images
    img_before = cv2.imread(os.path.join(input_folder, 'before.jpg'))
    img_after = cv2.imread(os.path.join(input_folder, 'after.jpg'))

    if img_before is None or img_after is None:
        raise ValueError("Images not found or path is incorrect.")

    # ---
    # Align images
    aligned_before = img_before # align_image(img_before)  # Change later
    aligned_after = img_after # align_image(img_after)  # Change later


    # Create a mask for the pink circle
    hsv = cv2.cvtColor(img_before, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])  # Adjust these values based on the image
    upper_pink = np.array([170, 255, 255])  # Adjust these values based on the image

    # calibrates mask
    while(1):
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        debug_print('pink mask', pink_mask)
        k = cv2.waitKey(0)
        print(k)
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == 119: # w
            lower_pink = lower_pink + [1, 0, 0]
        if k == 115: # s
            lower_pink = lower_pink + [-1, 0, 0]
        if k == 97: # a
            upper_pink = upper_pink + [-1, 0, 0]
        if k == 100: # d
            upper_pink = upper_pink + [1, 0, 0]


    debug_print('pink mask', cv2.bitwise_not(pink_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract yellow regions
    lower_yellow = np.array([20, 100, 100])  # Adjust these values based on the image
    upper_yellow = np.array([30, 255, 255])  # Adjust these values based on the image

    yellow_after = extract_color_region(aligned_after, lower_yellow, upper_yellow, pink_mask)


    # calibrates before
    while(1):
        yellow_before = extract_color_region(aligned_before, lower_yellow, upper_yellow, pink_mask)
        debug_print('yellow before', yellow_before)
        k = cv2.waitKey(0)
        print(k)
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == 119: # w
            lower_yellow = lower_yellow + [1, 0, 0]
        if k == 115: # s
            lower_yellow = lower_yellow + [-1, 0, 0]
        if k == 97: # a
            upper_yellow = upper_yellow + [-1, 0, 0]
        if k == 100: # d
            upper_yellow = upper_yellow + [1, 0, 0]

    # Calculate areas
    area_before = calculate_area(yellow_before)
    area_after = calculate_area(yellow_after)
    growth = area_after - area_before
    growth_percentage = (growth / area_before) * 100 if area_before != 0 else float('inf')
    inaccuracy = abs(growth) / area_before * 100 if area_before != 0 else float('inf')

    # Save results to CSV
    results = {
        'Experiment ID': [experiment_id],
        'Area Before': [area_before],
        'Area After': [area_after],
        'Growth': [growth],
        'Growth Percentage': [growth_percentage],
        'Measurement Inaccuracy (%)': [inaccuracy]
    }

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)

    print(f"Results saved to {os.path.join(output_folder, 'results.csv')}")


def debug_print(window_name, img):
#For debugging
    ratio = 0.2
    size = (math.floor(3000 * ratio), math.floor(4000 * ratio))
    cv2.imshow('debug: ' + window_name, cv2.resize(img, size))
# ---

if __name__ == "__main__":
    experiment_id = input("Enter experiment ID: ")
    main(experiment_id)
