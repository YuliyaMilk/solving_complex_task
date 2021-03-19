import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def kmeans(samples, k, criteria, attempts, flags):
    it = 0

    #Y = np.random.rand(k, flags[1])
    cent = []
    for i in range(k):
        cent.append([])
        for j in range(flags[1]):
            cent[i].append(random.randrange(0, 256))
    Y = np.array(cent)
    while it < criteria:
        dist1 = samples - Y[:, np.newaxis]
        distances = np.sqrt((dist1**2).sum(axis = 2))
        closest_centroids = np.argmin(distances, axis = 0)
        NEW_CENTERS = np.zeros((k, flags[1]))
        COUNT_OF_POINTS = np.zeros(k)
        for i, c in enumerate(closest_centroids):
            NEW_CENTERS[c] += samples[i]
            COUNT_OF_POINTS[c] += 1

        for i in range(k):
            if(COUNT_OF_POINTS[i] != 0):
                NEW_CENTERS[i] /= COUNT_OF_POINTS[i]
            else:
                NEW_CENTERS[i] = Y[i]
        Y = NEW_CENTERS.copy()
        it += 1
    dist1 = samples - Y[:, np.newaxis]
    distances = np.sqrt((dist1**2).sum(axis = 2))
    closest_centroids = np.argmin(distances, axis = 0)

    return (closest_centroids,Y)

image = cv2.imread("demonstration-image.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))

pixel_values = np.float32(pixel_values)

criteria = 10
criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3

labels, centers = kmeans(pixel_values, k, criteria, 10, (pixel_values.shape[0], pixel_values.shape[1]))
_, labels2, (centers2) = cv2.kmeans(pixel_values, k, None, criteria2, 10, cv2.KMEANS_RANDOM_CENTERS)

print(centers, centers2)

centers = np.uint8(centers)
centers2 = np.uint8(centers2)

segmented_image = centers[labels]
segmented_image2 = centers2[labels2.flatten()]

segmented_image = segmented_image.reshape(image.shape)
segmented_image2 = segmented_image2.reshape(image.shape)
# show the image
plt.imshow(segmented_image)
plt.show()
# show the image
plt.imshow(segmented_image2)
plt.show()

