import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
image = np.array([
    [1,0,1,0,1,0],
    [0,1,1,0,1,1],
    [1,0,1,0,1,0],
    [1,0,1,1,1,0],
    [0,1,1,0,1,1],
    [1,0,1,0,1,0]
])
kernel = np.array([
    [1,0,1],
    [0,1,0],
    [1,0,1]
])
print(image)
def convolve(image,kernel):
    k = kernel.shape[0]
    outsize = image.shape[0] - k + 1
    result = np.zeros((outsize,outsize))
    for i in range(outsize):
        for j in range(outsize):
            patch = image[i:i+k,j:j+k]
            result[i,j] = np.sum(patch*kernel)
    return result
featured_map = convolve(image,kernel)
print(kernel)
print(featured_map)

def maxpool(festured_map, size = 2):
        outsize = featured_map.shape[0]//2
        result = np.zeros((outsize,outsize))
        for i in range(outsize):
             for j in range(outsize):
                  window = featured_map[i * size: (i+1) * size,j * size:(j+1) * size]
                  result[i,j] = np.max(window)
        return result
pool = maxpool(featured_map,size = 2)
print(pool)

flatter = pool.flatten()
print(flatter)

model = Sequential()
model.add(Input(shape = (6,6,1)))
model.add(Conv2D(4,kernel_size = (3,3),activation = "relu"))
print(f"after conve2d {model.output_shape}")