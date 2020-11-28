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

driver.get('https://www.amazon.com/dp/B082LNTGYP/ref=s9_acsd_simh_ft_c2_x_3_i?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=center-10&pf_rd_r=KD26PZ2HZFP0QJNKGZBQ&pf_rd_t=1401&pf_rd_p=6e914f55-e694-49f1-a0db-fba765b584aa&pf_rd_i=1000844301')
time.sleep(3)

#find the clickable image that opens the book
driver.find_element_by_id('ebooksImgBlkFront').click()
time.sleep(3) 

#go to the print tab
driver.find_element_by_id('readerModeTabPrint').click()

#create set of images   
imageList = set()

time.sleep(10)


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





#tess data train source
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

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)



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
    file = open('/home/jed/Desktop/pageN.txt', 'w+')
    file.write(ocr_core('page.png'))
    f = open('/home/jed/Desktop/pageN.txt', 'r')
    print(f.read())



