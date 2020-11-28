from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from urllib.request import urlretrieve
import subprocess
import cv2
import numpy as np
import pytesseract
from PIL import Image
import shutil



#initiate chrome driver
driver = webdriver.Chrome()

driver.get('https://www.amazon.com/War-Peace-Leo-Nikolayevich-Tolstoy/dp/1427030200')
time.sleep(3)

#find the clickable image that opens the book
driver.find_element_by_id('img-canvas').click()

#create set of images
imageList = set()

time.sleep(10)
print (driver.find_element_by_id('sitbReaderRightPageTurner').get_attribute('style'))

#while pointer is on display then run loop
while 'pointer' in driver.find_element_by_id('sitbReaderRightPageTurner').get_attribute('style'):
    #click on the right arrow and move through the pages
    driver.find_element_by_id('sitbReaderRightPageTurner').click()
    time.sleep(3)

    #get any new page that has loaded(multiple pages can load)
    pages = driver.find_elements_by_xpath("//div[@class='pageImage']/div/img")
    # pages = driver.find_elements_by_select("div[class='pageImage'] > div > img")
    for page in pages:
        image = page.get_attribute('src')
        imageList.add(image)
        # print (imageList)
        # print ('done')

driver.quit()

# #start processing the images collected through the urls
# for image in sorted(imageList):
#     urlretrieve(image, 'page.jpg')
#     p = subprocess.Popen(['tesseract', 'page.jpg', '/home/jed/Desktop/page'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     p.wait()

#     f = open('/home/jed/Desktop/page.txt', 'r')
#     print(f.read())
#     # file = open('/home/jed/Desktop/pageNew.txt', 'w+')
#     # file.write(f)

#     shutil.copy('/home/jed/Desktop/page.txt', '/home/jed/Desktop/pageNew.txt')




#image
img = cv2.imread('image.jpg')

def t_tesseract():
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

def resize(image):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    scale_percent = 220 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

 #ocr definition   
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    t_tesseract()
    cv = resize(filename)
    gray = get_grayscale(cv)
    thresh = thresholding(gray)
    openn = opening(gray)
    text = pytesseract.image_to_string(openn, lang='eng')  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text


#start processing the images collected through the urls
for image in sorted(imageList):
    pp = urlretrieve(image, 'page.png')
    file = open('/home/jed/Desktop/page.txt', 'w+')
    file.write(ocr_core('page.png'))
    f = open('/home/jed/Desktop/page.txt', 'r')
    print(f.read())