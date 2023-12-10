# import the necessary packages
from collections import deque

import numpy as np
import argparse
import cv2
import imutils
import time
from PIL import Image
import mss

import numpy
import pyautogui
print("1")
i=0
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

bgr_color = np.uint8([[[25, 68, 178]]])  # 这是一个红色到绿色的过渡色，你可以替换成其他颜色
hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

# 获取 HSV 颜色范围
greenLower = np.array([hsv_color[0][0][0] - 10, 10, 185])  # 下限颜色值
greenUpper = np.array([hsv_color[0][0][0] + 50, 120, 250])  # 上限颜色值

bgr_color = np.uint8([[[0, 150, 150]]])  # 这是一个红色到绿色的过渡色，你可以替换成其他颜色
hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

# 获取 HSV 颜色范围
blueLower = np.array([30,30,200])  # 下限颜色值
blueUpper = np.array([150,100,255])  # 上限颜色值

pts = deque(maxlen=args["buffer"])

with mss.mss() as sct:
    monitor = {"top": 540, "left": 828, "width": 257, "height": 80}
    while "Screen capturing":
            
            # last_time = time.time()
            # vs = numpy.array(sct.grab(monitor))
            # print("fps: {}".format(1 / (time.time() - last_time)))
            vs = sct.grab(monitor)
            frame = np.array(vs)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv_imge = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_imge, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)

            mask2 = cv2.inRange(hsv_imge, blueLower, blueUpper)
            mask2 = cv2.erode(mask2, None, iterations=1)
            mask2 = cv2.dilate(mask2, None, iterations=1)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = imutils.grab_contours(cnts2)
            center2 = None 
            
            print(i)
            
            if len(cnts) > 0 and len(cnts2) >0:
                
                c = max(cnts, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #計算中心點

                c2 = max(cnts2, key=cv2.contourArea)
                (x2, y2, w2, h2) = cv2.boundingRect(c2)
                M2 = cv2.moments(c2)
                center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

                # draw the rectangle and centroid on the frame,
                # then update the list of tracked points

                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (int(x2), int(y2)), (int(x2+w2), int(y2+h2)),(0, 255, 255), 2)
                cv2.circle(frame, center2, 5, (0, 0, 255), -1)

                # update the points queue
                print(center)
                pts.appendleft(center)

                if center is not None and center[0]>144 or center[0]<144 :
                    i=i+1
                    print('clike')
                    pyautogui.mouseDown()
                    time.sleep(1)
                    pyautogui.mouseUp()
                    
            if center is None and i>0:
                    time.sleep(1)
                    pyautogui.mouseDown()
                    time.sleep(3)
                    pyautogui.mouseUp()
                    i = 0
            cv2.namedWindow('Window Title', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Window Title', frame)
            cv2.setWindowProperty('Window Title', cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(1)
            if key == ord("q"):
                        cv2.destroyAllWindows()
                        break
