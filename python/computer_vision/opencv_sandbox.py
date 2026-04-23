import cv2
import matplotlib.pyplot as plt


# Simple cv2 usage

image = cv2.imread('image.png')

cv2.imshow("Autostereogram", image)
cv2.waitKey(0) # Waits indefinetly until a key is pressed.
cv2.destroyAllWindows() # Closes windows opened by cv2

# ------------------------------------------------------- #

# Loads image and shows with cvtColor

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Exhibits image
plt.imshow(image_rgb)
# plt.axis('off') # Desabilita os eixos
plt.show()

# ------------------------------------------------------- #

# Loads image and shows with grayscale conversion

image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image_grayscale, cmap='gray')
plt.show()

# ------------------------------------------------------- #

# GaussianBlur - Image Soften

softened_image = cv2.GaussianBlur(image, (15, 15), 0)

softened_image_rgb = cv2.cvtColor(softened_image, cv2.COLOR_BGR2RGB)

plt.imshow(softened_image_rgb)
plt.show()

# ------------------------------------------------------- #

# Canny - Border Detection

# Use grayscale transformed image
borders = cv2.Canny(image_grayscale, 100, 200)

plt.imshow(borders, cmap='gray')
plt.show()