import cv2
import pytesseract
image = cv2.imread('keepers_diary.png')

cv2.imshow('Loaded Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Test with pytesseract: ", pytesseract.image_to_string(image=image))