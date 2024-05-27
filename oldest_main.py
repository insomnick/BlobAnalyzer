import numpy as np
import cv2

print('Hello Wilfried')

# path
path = r'./data/blob_frei.JPEG'
image = cv2.imread(path)

#Ab hier nur Darstellung
#Nicht notwendig
h, w, c = image.shape
if h > w:   # Rotieren falls hochkant
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Aruco detect
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(image)
cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

#Pink Circle Recognition
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_pink = np.array([130, 0, 180])
upper_pink = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower_pink, upper_pink)


window_name = 'Blob'
scale = 0.4
resized_down = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
cv2.startWindowThread()
while(1):
    cv2.imshow(window_name, resized_down)
    k = cv2.waitKey(0)
    print(k)
    if k == 27: break
    #up == 82
    #down == 84
    #left == 81
    #right == 83


cv2.destroyAllWindows()
