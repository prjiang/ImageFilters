# 測試各種 模糊遮罩
import sys
import cv2 as cv
import numpy as np

# Get your pic for test
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    _, img = cap.read()
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    '''
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)    
    

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    '''
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    
    # save pic and quit VideoCapture
    if cv.waitKey(90) & 0xFF == ord('s'):
        cv.imwrite('ai/cvtest.jpg', gray)
        cv.imwrite('ai/cvtest_color.jpg', img)
    if cv.waitKey(90) & 0xFF == ord('q'):
        break

cap.release() 
cv.destroyAllWindows() 

#  Global Variables
DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31
src = None
dst = None
window_name = 'Smoothing Demo'
def main(argv):
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    # Load the source image
    imageName = argv[0] if len(argv) > 0 else 'ai/cvtest.jpg'
    global src
    src = cv.imread(cv.samples.findFile(imageName))
    if src is None:
        print ('Error opening image')
        print ('Usage: smoothing.py [image_name -- default ../data/cvtest.jpg] \n')
        return -1
    if display_caption('Original Image') != 0:
        return 0
    global dst
    dst = np.copy(src)
    if display_dst(DELAY_CAPTION) != 0:
        return 0

    cv.waitKey()    
    # Applying Homogeneous blur
    if display_caption('Homogeneous Blur') != 0:
        return 0
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv.blur(src, (i, i))
        if display_dst(DELAY_BLUR) != 0:
            return 0
    cv.waitKey()
    # Applying Gaussian blur
    if display_caption('Gaussian Blur') != 0:
        return 0
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv.GaussianBlur(src, (i, i), 0)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    cv.waitKey()
    # Applying Median blur
    if display_caption('Median Blur') != 0:
        return 0
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv.medianBlur(src, i)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    cv.waitKey()
    # Applying Bilateral Filter
    if display_caption('Bilateral Blur') != 0:
        return 0
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv.bilateralFilter(src, i, i * 2, i / 2)
        if display_dst(DELAY_BLUR) != 0:
            return 0
    cv.waitKey()
    #  Done
    display_caption('Done!')
    return 0
def display_caption(caption):
    global dst
    dst = np.zeros(src.shape, src.dtype)
    rows, cols, _ch = src.shape
    cv.putText(dst, caption,
                (int(cols / 4), int(rows / 2)),
                cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    return display_dst(DELAY_CAPTION)
def display_dst(delay):
    cv.imshow(window_name, dst)
    c = cv.waitKey(delay)
    if c >= 0 : return -1
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])