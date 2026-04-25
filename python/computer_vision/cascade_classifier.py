import cv2

image = cv2.imread('stars.png')

cv2.imshow('Loaded Image - No Processing', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Haarcascade model, at least the one we're using, requires a grayscale to work.

grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


cv2.imshow('Loaded Image - Grayscale', grayscaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

faces = face_cascade.detectMultiScale(grayscaled, scaleFactor=1.1, minNeighbors=17)

#Draw Shapes around the detected faces

for (x, y, w, h) in faces:
    center = (x + w//2, y + h//2)   # center of the face
    radius = w//2                   # approximate radius
    cv2.circle(image, center, radius, (0, 255, 0), 2)


cv2.imshow('Loaded Image - Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
