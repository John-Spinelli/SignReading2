import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
import skimage as sk
import csv

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\johnn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

NUM_SAMPLE = 12
#match_ratio = 0.70 #.75
#test_img = 'test_8.png'
#test_img = 'sign_13.png'


def remove_noise(bin_im):
    height, width = bin_im.shape

    for row in range(1,height-1):
        for col in range(1, width-1):
            neighbours = bin_im[row-1,col] + bin_im[row+1,col]
            neighbours += bin_im[row,col-1] + bin_im[row,col+1]
            neighbours += bin_im[row+1,col-1] + bin_im[row+1,col+1]
            neighbours += bin_im[row-1,col-1] + bin_im[row-1,col+1]
            if neighbours < 2 and bin_im[row,col] >= 1:
                bin_im[row,col] = 0
    return bin_im

def hog_crop(image):
    height_o, width_o, _ = image.shape
    image_blur = cv2.blur(image, (3,3))
    resized_img = sk.transform.resize(image_blur, (128*4, 64*4))

    #cv2.imshow('image', image)
    #cv2.imshow('resized', resized_img)
    fd, hog_image = sk.feature.hog(resized_img, orientations=9,
                                   pixels_per_cell=(8,8), cells_per_block=(2,2),
                                   visualize=True, multichannel=True)

    #plt.imshow(hog_image, cmap='gray')
    #plt.show()

    #print(np.amax(hog_image))
    #print(hog_image)
    hog_image = hog_image * 255
    #print(hog_image)

    ret,thresh2 = cv2.threshold(hog_image,(np.amax(hog_image))*0.5,255,cv2.THRESH_BINARY)
    thresh2 = remove_noise(thresh2)
    #####cv2.imshow('result?',thresh2)

    bigboys = sk.transform.resize(thresh2, (height_o, width_o))
    #cv2.imshow('back to big', bigboys)

    coords = cv2.findNonZero(bigboys)
    x,y,w,h = cv2.boundingRect(coords)
    c_w = int(w*0.1)
    c_h = int(h*0.1)
    left = x-c_w
    right = x+w+c_w
    top = y-c_h
    bot = y+h+c_h
    if left < 0:
        left = 0
    if right >= width_o:
        right = width_o-1
    if top < 0:
        top = 0
    if bot >= height_o:
        bot = height_o-1
    result = image[top:bot, left:right]
    ######cv2.imshow('hogresult',result)

    '''
    cv2.imshow('image', image)
    cv2.imshow('bigboys', bigboys)
    cv2.imshow('result', result)

    cv2.imwrite('_HOG_before.png', image)
    cv2.imwrite('_HOG_small.png', bigboys)
    cv2.imwrite('_HOG_large.png', result)
    '''
    return result




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
            ########cv2.imshow('after '+str(i), base)

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

    #######cv2.imshow('mask',mask)
    return mask        

# Preprocessing of given image
def preproc(img):
    ''' Return smoothed image, canny edges and grayscale '''
    blur = cv2.blur(img, (5,5))
    #cv2.imshow('blurred',blur)

    edges = cv2.Canny(blur, 140,200)
    #cv2.imshow('canny',edges)

    return blur, edges


def getMatches(img1, img2, match_ratio):
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
    #cv2.imshow('matched_img',matched_img)
    cv2.imwrite('_SIFT_matches.png',matched_img)
    ''' End Display '''
    return

def matchImages(img1, img2, match_ratio):
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
    #kernel = np.ones((5,5),np.uint8)
    #closed = cv2.morphologyEx(extracted, cv2.MORPH_CLOSE, kernel)
    #closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    return extracted

def readSign(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh2 = cv2.threshold(g,120,255,cv2.THRESH_BINARY)
    #cv2.imshow('threshed',thresh2)

    kernel = np.ones((7,7),np.uint8)
    closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    thresh2 = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    ###########cv2.imshow('about to read', thresh2)

    #print('===========')
    #print(pytesseract.image_to_string(thresh2, config='--psm 9'))

    syms = '\n ,./<>?;:"'
    lower = 'qwertyuiopasdfghjklzxcvbnm'
    upper = lower.upper()
    strip_chars = syms+lower+upper
    
    text = pytesseract.image_to_string(thresh2, config='--psm 9')
    #text = text.strip(strip_chars)
    text = text.strip(strip_chars)
    # Correct instances of ]}) as a 1
    for index in range(len(text)):
        if text[index] in ')}]':
            text = text[0:index] + '1' + text[index+1:]
        
    #print(text)
    
    #for i in [6,7,9,10]:
        #print('===========')
        
        #text = pytesseract.image_to_string(thresh2,config='--psm '+str(i))
        #try:
            #text = text.strip(strip_chars)
            #print(text)
            #print(i)
        #except:
        #    pass

    return text, thresh2


def removeBackground(img):
    ''' Attempts to remove foliage or sky from the image to remove any
    features that could impact SIFT '''
    t1 = cv2.medianBlur(img,5)
    t1 = cv2.blur(t1, (5,5))
    #t1 = cv2.blur(img, (5,5))
    
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

    remove = cv2.inRange(t1, lower, upper)
    remove = cv2.bitwise_not(remove)
    kernel = np.ones((5,5),np.uint8)
    remove = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel)

    final = cv2.bitwise_and(img, img, mask=remove)
    """
    cv2.imshow('img',img)
    cv2.imshow('final',final)
    cv2.imwrite('_HSV_before.png',img)
    cv2.imwrite('_HSV_after.png',final)
    """
    #final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    ########cv2.imshow('background removed',final)
    return final

####################################
####################################
####################################

def run_test(match_thresh):

    results = []
    
    for i in range(2,3):  # 1,28
        test_img = 'sign_' + str(i) + '.png'
        #if i > 13:
        #    test_img = 'test_' + str(i-13) + '.png'
        try:
            image = cv2.imread('blank.png')
            #####cv2.imshow('blank',image)
            b_im, e_im = preproc(image)
            mask = getMask(e_im)

            test = cv2.imread(test_img)
            test = hog_crop(test)
            #####readSign(test)
            #print('++++++++++++++++')
            test = removeBackground(test)

            cv2.imwrite('_SIFT_before.png',test)

            #b_test, e_test = preproc(test)
            getMatches(test, image, match_thresh)######
            #result = matchImages(b_test, b_im)
            result = matchImages(test, image, match_thresh)
            #cv2.imshow("result",result) ##########
            #cv2.imwrite('_SIFT_1.png',result)

            try:
                prev = result
                result = matchImages(result, image, match_thresh)
                #cv2.imwrite('_SIFT_2.png',result)
                #####cv2.imshow("result2",result)
                
                result = matchImages(result, image, match_thresh)
                #cv2.imwrite('_SIFT_3.png',result)
                #cv2.imshow("result3",result)
                
                #result = matchImages(result, image, match_thresh)
                #cv2.imshow("result4",result)
                cv2.imwrite("_Isolated_before.png",result)
                result = isolateSign(result, mask)
                #cv2.imwrite('isolated_'+str(i)+'.png', result)
                #####cv2.imshow("Isolated",result)
                cv2.imwrite("_Isolated.png",result)
                
                text, fin_img = readSign(result)
                results.append(text)
                cv2.imwrite('binary_'+str(i)+'.png', fin_img)
            except:
                #####print("something else went wrong, iteration", i)
                result = isolateSign(prev, mask)
                cv2.imwrite('isolated_'+str(i)+'.png', result)
                
                text, fin_img = readSign(prev)
                results.append(text)
                cv2.imwrite('binary_'+str(i)+'.png', fin_img)
                
        except:
            #######print("Failed on First attempt of", i)
            results.append("Fail")
            
        cv2.waitKey(0)    

    return results

'''###### Testing Zone #######'''
expected  = ['69','7','83','9','2','70','17','17','11','69','7','2','401','105','12','401','400','401','64','401','400','96','7','400','35','3','89']
test_res = []

best_ratio = 0.8

#for i in range(0,15):
test_res.append( run_test(best_ratio) )

#print(expected)
#print(test_res)

"""

with open('TEST_RESULTS_bestmatch','w') as f:
    write = csv.writer(f)
    write.writerow(expected)
    for row in test_res:
        write.writerow(row)
"""

print("/////////////////////////////")
print("/////////////////////////////")
print("/////////////////////////////")
print("/////////////////////////////")

# From test results, using match ratio of 0.8, 0.7 or 0.69 yielded best accuracy

'''###########################'''

"""
s_1  - good, easy
s_2  - not enough matches
s_3  - good, easy
s_4  - no text detected
s_5  - bad warp
s_6  - bad warp
s_7  - bad warp, colour removed
s_8  - good, despite noise
s_9  - bad, i instead of 11
s_10 - good
s_11 - good
s_12 - bad warp, not enough matches. crown is also bad
s_13 - close, 40) instead of 401

t_1 - good
t_2 - bad warp, crop too low
t_3 - close, 40) instead of 401 >> same as s_13
t_4 - bad warp, other sign messes up
t_5 - bad warp, warps to other sign
t_6 - bad warp, foliage
t_7 - close, 40) instead of 401 >> same as t_5
t_8 - fail
"""


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
