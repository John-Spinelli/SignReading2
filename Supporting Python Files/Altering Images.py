import cv2
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

inputs = ['sign_1.png','sign_3.png','sign_8.png','sign_14.png',
          'sign_16.png','sign_17.png','sign_20.png','sign_24.png',
          'sign_26.png']

count = 28
for image in inputs:
    src_img = cv2.imread(image)
    img = src_img + (40,40,40)
    cv2.imwrite('sign_'+str(count)+'.png',img)
    count += 1
    img = src_img - (80,80,80)
    cv2.imwrite('sign_'+str(count)+'.png',img)
    count += 1

    img = rotate_image(src_img, 25)
    cv2.imwrite('sign_'+str(count)+'.png',img)
    count += 1

