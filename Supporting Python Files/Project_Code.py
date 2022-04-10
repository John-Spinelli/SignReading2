import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\johnn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

NUM_SAMPLE = 12
match_ratio = 0.72 #.75
test_img = 'test_1.png'
test_img = 'sign_6.png'

"""
sign_1: 6,7,9,10  
sign_2:          >>> Fail
sign_3: 6,7,9,10  >> ??? 
sign_4:           >> Fail
sign_6:           >> Fail
sign_7: 6,7,9,10
"""

def getPrototype():
    count = 1
    template = cv2.imread('blank.png')
    #base = cv2.imread('blank2.png')
    #template, t_edges = preproc(template)
    base = template
    
    
    for i in range(1,NUM_SAMPLE+1):
        try:
            if not(i==1 or i==3 or i==4 or i==5):
                continue
            print(i)
            layer = cv2.imread('sign_'+str(i)+'.png')
            #layer, l_edges = preproc(layer)
            getCannyMatches(layer, template)
            #layer = matchImages(layer, template)
            layer = cannyMatch(layer,template)
            base = cv2.addWeighted(base, count/(count+1), layer, 1/(count+1), 0)
            cv2.imshow('after '+str(i), base)

            cv2.waitKey(0)
            count += 1
        except:
            print("img "+str(i)+" failed")

    return base
'''
################ FUNCTIONS ################
'''

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
    cv2.drawContours(mask,[max_cont],0,255,-1,)

    cv2.imshow('mask',mask)
    return mask        

# Preprocessing of given image
def preproc(img):
    ''' Return smoothed image, canny edges and grayscale '''
    blur = cv2.blur(img, (5,5))
    cv2.imshow('blurred',blur)

    edges = cv2.Canny(blur, 140,200)
    cv2.imshow('canny',edges)

    return blur, edges


def getMatches(img1, img2):
    ''' This function just returns the number of matches for a pair
    of images. Use it to find best pair to combine '''
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
    ''' End Display '''
    return

def matchImages(img1, img2):
    ''' This function computes the homography to transform img1
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
    ''' returns just the sign from transformed image '''
    extracted = cv2.bitwise_and(image,image, mask = mask)
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(extracted, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    return closed

def readSign(img):
    g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    ret,thresh2 = cv2.threshold(g,120,255,cv2.THRESH_BINARY)
    cv2.imshow('threshed',thresh2)

    print('===========')
    #print(pytesseract.image_to_string(thresh2, config='--psm 9'))

    text = pytesseract.image_to_string(thresh2, config='--psm 9')
    text = text.strip('\n ,./<>?;:"[]{}()')
    print(text)

    
    for i in [6,7,9,10]:
        print('===========')
        
        text = pytesseract.image_to_string(thresh2,config='--psm '+str(i))
        try:
            text = text.strip('\n ,./<>?;:"[]{}()')
            print(text)
            print(i)
        except:
            pass


def removeBackground(img):
    ''' Attempts to remove foliage or sky from the image to remove any
    features that could impact SIFT '''
    t1 = cv2.medianBlur(img,5)
    t1 = cv2.blur(t1, (5,5))
    
    t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)

    # Values acquired by taking typical sky and foliage colours
    # from sample and test images
    LH = 0
    LS = 45
    LV = 30
    HH = 255
    HS = 255
    HV = 255

    lower = np.array([LH, LS, LV])
    upper = np.array([HH, HS, HV])

    remove = cv2.inRange(t1, lower, upper)
    remove = cv2.bitwise_not(remove)
    kernel = np.ones((5,5),np.uint8)
    remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel)

    final = cv2.bitwise_and(img, img, mask=remove)
    #final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    cv2.imshow('background removed',final)
    return final

try:
    image = cv2.imread('blank.png')
    cv2.imshow('blank',image)
    b_im, e_im = preproc(image)
    mask = getMask(e_im)

    test = cv2.imread(test_img)
    test = removeBackground(test)
    
    b_test, e_test = preproc(test)
    getMatches(test, image)
    #result = matchImages(b_test, b_im)
    result = matchImages(test, image)
    cv2.imshow("result",result)
except:
    print("Failed on First attempt")

try:
    result = matchImages(result, image)
    cv2.imshow("result2",result)

    result = matchImages(result, image)
    cv2.imshow("result3",result)

    result = matchImages(result, image)
    cv2.imshow("result4",result)

    #print('===========')
    #print(pytesseract.image_to_string(result))

    result = isolateSign(result, mask)
    cv2.imshow("Isolated",result)

    cv2.imshow("closed",result)
    readSign(result)
except:
    print("something else went wrong")

#g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#ret,thresh2 = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
#cv2.imshow('threshed',thresh2)

#thresh2 = cv2.Canny(thresh2, 50, 180)
#cv2.imshow('Canny',thresh2)

#print('===========')
#print(pytesseract.image_to_string(thresh2, config='--psm 6'))

'''
for i in range(3,14):
    print('===========')
    
    x = pytesseract.image_to_string(thresh2,config='--psm '+str(i))
    try:
        print(x)
        print(i)
    except:
        pass
'''


#print('===========')
#testtext = cv2.imread('testtext.png')
#print(pytesseract.image_to_string(testtext))


"""
prototype = getPrototype()
cv2.imshow("Prototype", prototype)
"""
'''
Structure:

take sign image, preprocess and use as template
> LP filter
> get edges for edge comparison 

take input image, preprocess
> LP filter
> get edges for edge comparison

Run SIFT to get the matches with the template sign

Transform image to be aligned with camera

get mask from edges and erase everything but the mask content

run tesseract on the resulting image


'''



cv2.waitKey(0)
cv2.destroyAllWindows()


"""
def getCannyMatches(img1, img2):
    ''' This function just returns the number of matches for a pair
    of images. Use it to find best pair to combine '''

    img1 = cv2.Canny(img1, 50, 180)
    img2 = cv2.Canny(img2, 50, 180)
    
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
        if m.distance < 0.9*n.distance:
            good_match.append([m])  

    ''' Just for display purposes, in case you want to visualize the matches '''
    matched_img = cv2.drawMatchesKnn(img1, keys1, img2, keys2, good_match, None, flags = 2)
    cv2.imshow('matched_img',matched_img)
    ''' End Display '''
    return


def cannyMatch(img1, img2):
    ''' This function computes the homography to transform img1
        to match with img2. Returns the transformed img1 '''

    # Get grayscale images
    #img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_g = cv2.Canny(img1, 50, 180)
    img2_g = cv2.Canny(img2, 50, 180)
    
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
        if m.distance < 0.9*n.distance:
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
    """
