import keras_ocr
import pytesseract
import cv2
import numpy as np
# pipeline = keras_ocr.pipeline.Pipeline()
            
image='Crop/crop_1657523896159.jpg'
# image='download.jpeg'

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# im = cv2.filter2D(image, -1, kernel)
# print("im", type(im))
imag = cv2.imread(image)
edges = cv2.Canny(imag,100,200)
cv2.imshow("edge", edges)
gaussian_blur = cv2.GaussianBlur(imag, (7,7), 2)
sharpened1 = cv2.addWeighted(imag, 1.5, gaussian_blur, -0.5, 0)
sharpened2 = cv2.addWeighted(imag, 1.5, gaussian_blur, -2.5, 0)
sharpened3 = cv2.addWeighted(imag, 1.5, gaussian_blur, -6.5, 0)
cv2.imshow("sharpened1", sharpened1)
# cv2.imshow("sharpened2", sharpened2)
# cv2.imshow("sharpened3", sharpened3)
cv2.waitKey(0)
# cv2.destroyallwindow()
# cv2.release()

# images = [
#     keras_ocr.tools.read(image) 
# ]
# prediction_groups = pipeline.recognize(images)

# for i in prediction_groups:
#     for text, box in i:
#         print("text :::::::::", text)
        
# print("Tesseract OCR print:\n ",pytesseract.image_to_string(im, lang='eng'))