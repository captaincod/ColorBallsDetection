import cv2
import time
import numpy as np

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("WB", cv2.WINDOW_KEEPRATIO)

colors = {
    'green': [[65, 190, 80], [120, 255, 255]]
}

while cam.isOpened():
    _, image = cam.read()
    curr_time = time.time()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    green_lower = np.array(colors['green'][0])
    green_upper = np.array(colors['green'][1])
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_green = cv2.erode(mask_green, None, iterations=2)
    mask_green = cv2.dilate(mask_green, None, iterations=2)

    dist = cv2.distanceTransform(mask_green, cv2.DIST_L2, 5)
    reg, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(mask_green, fg)
    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    wmarkers = cv2.watershed(image, markers.copy())
    contours, hierarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (0, 255, 0), 6)
            cv2.putText(image, str(int((len(hierarchy[0])-1)/2)), (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0))

    cv2.imshow("Camera", image)
    cv2.imshow("WB", fg)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()