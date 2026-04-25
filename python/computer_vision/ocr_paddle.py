import os
import cv2
import matplotlib.pyplot as plot
import paddle
import numpy as np
paddle.set_flags({'FLAGS_use_mkldnn': False})
from paddleocr import PaddleOCR

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

def ocr_read_image(image):
    extract_ocr = ocr.ocr(image)
    result = ''
    for line in extract_ocr:
        for word in line:
            result += word[1][0] + ' '
        result += '\n'
    return result

ocr = PaddleOCR(lang='en', use_angle_cls=True, rec_model_dir=None, det_model_dir=None,)
pre_processing_result = ocr_read_image(image)
print("Pre Processing Result: ", pre_processing_result)

upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Upscaled', upscaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
post_upscaling_result = ocr_read_image(upscaled)
print("Post Upscaling Result: ", post_upscaling_result)

gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
post_grayscale_result = ocr_read_image(thresh_color)


cv2.imshow('Grayscaled + Threshold', thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Post Upscaling + Grayscale + Threshold Result: ", post_grayscale_result)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

sharpened = cv2.filter2D(thresh_color, -1, kernel)
post_sharpen_result = ocr_read_image(sharpened)
cv2.imshow('GraysCaled + Threshold + Sharpen: ', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()





