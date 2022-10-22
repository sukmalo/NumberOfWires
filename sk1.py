import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from skimage.measure import label
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import morphology

def translation(image, vector): #перенос изображения по координатам
    translated = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            ny = y + vector[0]
            nx = x + vector[1]
            if ny < 0 or nx < 0 or ny >= image.shape[0] or nx >= image.shape[1]:
                continue
            translated[ny, nx] = image[y, x] 
    return translated

#mask1 = np.ones((3, 3))
mask = [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]]

def delation(image, struct = mask): #наращивание краев фигуры(сглаживание)
    result = np.zeros_like(image)
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            rlog = np.logical_and(image[y, x], struct)
            result[y - 1 : y + 2, x - 1 : x + 2] = np.logical_or(result[y - 1: y + 2, x - 1: x + 2], rlog)
    return result

def erosion(image, struct = mask): #убирание краев фигуры
    result = np.zeros_like(image)
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            sub = image[y - 1 : y + 2, x - 1 : x + 2]
            if np.all(sub == struct):
                result[y, x] = 1
    return result

def closing(image, struct = mask): #сглаживание
    return erosion(delation(image, struct), struct)

def opening(image, struct = mask):
    return delation(erosion(image, struct), struct)
 

image = np.load("C:\\progrpython\\img\\wires6.npy.txt")  
labeled = label(image)

for i in range(1, labeled.max() + 1):
    copy = np.zeros_like(image)
    copy[labeled == i] = 1
    erosed = binary_erosion(copy, mask)
    erose = label(erosed)
    if erose.max() == 0:
        print(f"№{i} - полностью разорван")
    elif erose.max() == 1:
        print(f"№{i} - частей", 0)
    elif erose.max() >= 2:
        print(f"№{i} - частей", erose.max())

erased = label(binary_erosion(image, mask))
        
plt.subplot(121)
plt.imshow(labeled)
plt.subplot(122)
plt.imshow(erased)
plt.show()