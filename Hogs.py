import numpy as np
import cv2
from skimage.feature import hog

def extract_hog_features(image_path, cell_size=(8, 8), block_size=(2, 2), nbins=9):

    # Extrakce HOG vlastností
    hog_features = hog(small_image, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2-Hys')

    return hog_features

# Nastavení cesty k obrázku
image_path = '1_bus_1_den_frame756.png'

# Extrakce HOG vlastností
hog_features = extract_hog_features(image_path)
print("HOG features:", np.shape(hog_features))
