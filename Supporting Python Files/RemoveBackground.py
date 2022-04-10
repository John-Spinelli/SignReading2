import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract


t1 = cv2.imread('test_1.png')
t2 = cv2.imread('test_2.png')
t3 = cv2.imread('test_3.png')
t4 = cv2.imread('test_4.png')
t5 = cv2.imread('test_5.png')
t6 = cv2.imread('test_6.png')
t7 = cv2.imread('test_7.png')

t1 = cv2.cvtColor(t3, cv2.COLOR_BGR2HSV)
t1 = cv2.medianBlur(t1,5)
#t1 = cv2.blur(t1, (5,5))

LH = 20
LS = 45
LV = 30
HH = 170
HS = 255
HV = 255

lower = np.array([LH, LS, LV])
upper = np.array([HH, HS, HV])

remove = cv2.inRange(t1, lower, upper)
remove = cv2.bitwise_not(remove)

final = cv2.bitwise_and(t1, t1, mask=remove)
final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
cv2.imshow('final',final)


'''
LH = 20
LS = 39
LV = 30
HH = 170
HS = 255
HV = 255
'''
