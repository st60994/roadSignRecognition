import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image):
    cv2.imshow(title, cv2.resize(image,[720, 480]))
    cv2.waitKey(0)

def sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def laplace(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)

def prewitt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    return cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

def scharr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

def apply_custom_filter(image, kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.filter2D(gray, -1, kernel)
    return filtered_image



if __name__ == "__main__":
    image_path = 'image_5.png'
    image = cv2.imread(image_path)

    sobel_image = sobel(image)
    canny_image = canny(image)
    laplace_image = laplace(image)
    prewitt_image = prewitt(image)
    scharr_image = scharr(image)

    custom_kernel = np.array([
        [-1,   1, -1],
        [-1,   4, 1],
        [-1, -1, 1]
    ])

    filtered_image = apply_custom_filter(image, custom_kernel)

    show_image('Original Image', image)
    show_image('Sobel Operator', sobel_image)
    show_image('Canny Edge Detector', canny_image)
    show_image('Laplace Operator', laplace_image)
    show_image('Prewitt Operator', prewitt_image)
    show_image('Scharr Operator', scharr_image)
    show_image('Filtered Image', filtered_image)

    cv2.destroyAllWindows()