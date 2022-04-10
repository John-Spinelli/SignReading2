'''
Johnathan Spinelli
spinellj, 400128075
COMPENG 4TN4 Project Submission
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
import skimage as sk
import csv
import warnings
warnings.filterwarnings("ignore")

NUM_SAMPLES = 54

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\johnn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def remove_noise(bin_im):
    ''' This function receives a binary image as input, ideally a gradient image. This function
    removes instances of single pixels with less than two neighbouring pixels. Reduces noise in the
    image, allows more focus on the primary focus on the image. Returns image '''
    
    height, width = bin_im.shape
    # Iterate through image
    for row in range(1,height-1):
        for col in range(1, width-1):
            neighbours = bin_im[row-1,col] + bin_im[row+1,col]
            neighbours += bin_im[row,col-1] + bin_im[row,col+1]
            neighbours += bin_im[row+1,col-1] + bin_im[row+1,col+1]
            neighbours += bin_im[row-1,col-1] + bin_im[row-1,col+1]
            # Remove isolated pixels
            if neighbours < 2 and bin_im[row,col] >= 1:
                bin_im[row,col] = 0
    return bin_im

def hog_crop(image):
    ''' Receives image, computes the gradient and attempts to crop the image to
    the area with high contrast. Ignores areas with low contrast, or isolated
    instances of high contrast. Returns the cropped version of source image '''
    height_o, width_o, _ = image.shape
    image_blur = cv2.blur(image, (3,3))
    resized_img = sk.transform.resize(image_blur, (128*4, 64*4))

    # Compute HoG / Gradient of image
    fd, hog_image = sk.feature.hog(resized_img, orientations=9,
                                   pixels_per_cell=(8,8), cells_per_block=(2,2),
                                   visualize=True, multichannel=True)

    # Scale up gradient values
    hog_image = hog_image * 255

    # Threshold and remove noise
    ret,thresh = cv2.threshold(hog_image,(np.amax(hog_image))*0.5,255,cv2.THRESH_BINARY)
    thresh = remove_noise(thresh)
    upscaled = sk.transform.resize(thresh, (height_o, width_o))

    # Crop image based on high contrast pixels. Include padding
    coords = cv2.findNonZero(upscaled)
    x,y,w,h = cv2.boundingRect(coords)
    c_w = int(w*0.1)
    c_h = int(h*0.1)
    left = x-c_w
    right = x+w+c_w
    top = y-c_h
    bot = y+h+c_h
    # Correct in case of padding overflow
    if left < 0:
        left = 0
    if right >= width_o:
        right = width_o-1
    if top < 0:
        top = 0
    if bot >= height_o:
        bot = height_o-1
        
    result = image[top:bot, left:right]
    return result


def getMask(img):
    ''' returns mask of the template image '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_a = None
    max_cont = None

    # Find the outermost contour
    for cont in contours:
        # First contour
        if max_a == None:        
            max_a = cv2.contourArea(cont)
            max_cont = cont
            
        elif cv2.contourArea(cont) > max_a:
            # Longer contour than before
            max_a = cv2.contourArea(cont)
            max_cont = cont

    # Create mask from contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask,[max_cont],0,255,-1,lineType = cv2.LINE_AA)
    return mask        


def preproc(img):
    ''' Return smoothed image, canny edges and grayscale '''
    blur = cv2.blur(img, (5,5))
    edges = cv2.Canny(blur, 140,200)
    return blur, edges


def getMatches(img1, img2, match_ratio):
    ''' Display the matches between img1 and img2, mainly used for debugging
    and visualizing how successful the transformation will be (No longer
    used in code, but kept for future use) '''
    # Create the SIFT detector
    sift = cv2.SIFT_create()
    keys1, desc1 = sift.detectAndCompute(img1,None)
    keys2, desc2 = sift.detectAndCompute(img2,None)

    # Brute Force Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Ratio Test
    good_match = []

    # Compare distances between two nearest points
    for m,n in matches:
        if m.distance < match_ratio*n.distance:
            good_match.append([m])  

    ''' Just for display purposes, in case you want to visualize the matches '''
    matched_img = cv2.drawMatchesKnn(img1, keys1, img2, keys2, good_match, None, flags = 2)
    cv2.imshow('matched_img',matched_img)
    return

def matchImages(img1, img2, match_ratio):
    ''' Compute the homography to transform img1
        to match with img2. Returns the transformed img1 '''

    # Get grayscale images
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    keys1, desc1 = sift.detectAndCompute(img1_g,None)
    keys2, desc2 = sift.detectAndCompute(img2_g,None)

    keys1 = np.float32([kp.pt for kp in keys1])
    keys2 = np.float32([kp.pt for kp in keys2])

    # Brute Force Matching
    bf = cv2.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(desc1, desc2, 2)

    # Ratio Test
    good_match = []
    # Compare distances between two nearest points
    for m,n in matches:
        if m.distance < match_ratio*n.distance:
            good_match.append((m.trainIdx,m.queryIdx))

    pts1 = np.float32([keys1[i] for (_, i) in good_match])
    pts2 = np.float32([keys2[i] for (i, _) in good_match])
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)

    # Warp img1 to its matching perspective
    result = cv2.warpPerspective(img1, H,
    	(img1.shape[1] + img2.shape[1], img1.shape[0]+img2.shape[0]))

    result = np.array(result)
    result = result[0:img2.shape[0],0:img2.shape[1]]
    res = imutils.resize(result, width = 400)
    
    return result

def isolateSign(image, mask):
    ''' Apply mask to image, return result '''
    extracted = cv2.bitwise_and(image,image, mask = mask)
    return extracted

def tesseract_text(img):
    ''' Use Tesseract to read the text from the input image.
    Strip symbols or other text that are not digits from the
    text. Corrects for 1 being read as ]}). Returns text'''

    # Define characters to strip
    syms = '\n ,./<>?;:"・~({[“°‘—_・|\\・-'
    lower = 'qwertyuiopasdfghjklzxcvbnm'
    upper = lower.upper()
    strip_chars = syms+lower+upper

    # Read and strip chars
    text = pytesseract.image_to_string(img, config='--psm 9')
    text = text.strip(strip_chars)
    
    # Correct instances of ]}) as a 1
    for index in range(len(text)):
        if text[index] in ')}]':
            text = text[0:index] + '1' + text[index+1:]
    return text

def catch_failed_text(text):
    ''' Returns empty string is text does not contain only
    digits, returns text if only digits '''
    for letter in text:
        if ord(letter) >= 58 or ord(letter) <= 47:
            return ''
    return text

def readSign(img):
    ''' Perform morphological operations to isolate text in the input image.
    Attempt to read text, perform further operations if the read text is invalid.
    Returns text and final image when read '''

    # Threshold grayscale image
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(g,120,255,cv2.THRESH_BINARY)

    kernel = np.ones((7,7),np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    # Apply rectangular mask to isolate text
    rect_mask = cv2.imread('mask_rectangle.png')
    rect_mask = cv2.cvtColor(rect_mask, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(closed,closed, mask=rect_mask)

    # Crop to mask
    coords = cv2.findNonZero(image)
    x,y,w,h = cv2.boundingRect(coords)
    image = image[y:y+h, x:x+w]

    # Read text from image
    text = tesseract_text(image)
    text = catch_failed_text(text)

    # If unsuccessful, try reading from contours
    if text == '':
        # Canny operation
        canny = cv2.Canny(image, 140,200)
        closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

        # Get and draw contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        image = np.zeros(closed.shape, np.uint8)
        for cont in contours:

            if cv2.arcLength(cont,True) > 20 and cv2.contourArea(cont) < 80000 and cv2.contourArea(cont) > 3000: 
                cv2.drawContours(image,[cont],0,255,-1,lineType = cv2.LINE_AA)
        # Read again
        text = tesseract_text(image)
        text = catch_failed_text(text)

    return text, image


def removeBackground(img):
    ''' Attempts to remove foliage or sky from the image through HSV to remove any
    features that could impact SIFT '''

    # Blur image to smooth colours
    t1 = cv2.medianBlur(img,5)
    t1 = cv2.blur(t1, (5,5))
    t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)

    # Values acquired by taking typical sky and foliage colours
    # from sample and test images
    LH = 0
    LS = 84
    LV = 30
    HH = 255
    HS = 255
    HV = 255

    lower = np.array([LH, LS, LV])
    upper = np.array([HH, HS, HV])

    # Threshold using range
    remove = cv2.inRange(t1, lower, upper)
    remove = cv2.bitwise_not(remove)
    kernel = np.ones((5,5),np.uint8)
    remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel)
    final = cv2.bitwise_and(img, img, mask=remove)
    
    return final

####################################
####################################
####################################

def run_test(match_thresh):
    ''' Runs through algorithm to try to read text from sign. Returns
    list of all read values '''
    results = []

    # Iterate through every sign file
    for i in range(1,NUM_SAMPLES+1):  
        test_img = 'sign_' + str(i) + '.png'

        try:
            # Try to match and transform image
            image = cv2.imread('blank.png')
            b_im, e_im = preproc(image)
            mask = getMask(e_im)

            test = cv2.imread(test_img)
            test = hog_crop(test)
            test = removeBackground(test)

            result = matchImages(test, image, match_thresh)

            try:
                # First transformation successful. Attempt to further transform
                prev = result
                result = matchImages(result, image, match_thresh)
                result = matchImages(result, image, match_thresh)

                # Read text from image and append text to list
                result = isolateSign(result, mask)
                cv2.imwrite('isolated_'+str(i)+'.png', result)
                
                text, fin_img = readSign(result)
                results.append(text)
                cv2.imwrite('binary_'+str(i)+'.png', fin_img)
            except:
                # Second or third transform failed, use first transformation result
                result = isolateSign(prev, mask)
                cv2.imwrite('isolated_'+str(i)+'.png', result)
                
                text, fin_img = readSign(prev)
                results.append(text)
                cv2.imwrite('binary_'+str(i)+'.png', fin_img)
                
        except:
            # First transformation failed, result will be unpredictable
            results.append("Fail")   

    return results

'''###### Testing Zone #######'''
# Create list for test number
test_num = []
for i in range(1,NUM_SAMPLES+1):
    test_num.append(str(i))

# Hardcoded expected results, in order
expected  = ['69','7','83','9','2','70','17','17','11','69','7','2','401','105','12','401','400','401','64','401','400','96','7','400','35','3','89',
             '69','69','69','83','83','83','17','17','17','105','105','105','401','401','401','400','400','400','401','401','401','400','400','400','3','3','3']

# Run algorithm to get the read text from signs
# From test results, using match ratio of 0.8 yielded best accuracy
test_res = []
best_ratio = 0.8
test_res.append( run_test(best_ratio) )


# Write out signs to file to review results
with open('TEST_RESULTS_bestmatch.txt','w') as f:
    write = csv.writer(f)
    write.writerow(test_num)
    write.writerow(expected)
    for row in test_res:
        write.writerow(row)

print("/////////////////////////////")
print("/////////////////////////////")
print("/////////////////////////////")
print("/////////////////////////////")


