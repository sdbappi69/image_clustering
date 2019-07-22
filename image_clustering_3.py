# import the necessary packages
import os
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import python_utils
import numpy as np

# For Animation
import time
import itertools
import threading
import sys

done = False
start_time = time.time()
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rProcess time: %s seconds\n' % (time.time() - start_time))

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def get_color_count(image, default_color_count = 6):
    color_count = 0
    unique_colors = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            b, g, r = image[x, y]
            c = rgb2hex(r,g,b)
            if c not in unique_colors:
                unique_colors.append(c)
    # print(unique_colors)
    if len(unique_colors) == 0:
        color_count = default_color_count
    else:
        color_count = len(unique_colors)

    return color_count

def get_clt(image, color_count = 6):
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = color_count)
    clt.fit(image)

    return clt

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

main_dir = os.path.split(os.path.abspath(__file__))[0]
file_name = 'color_plate.jpg'
image = cv2.imread(os.path.join(main_dir, file_name))

color_count = get_color_count(image)
print("Total color: " + str(color_count))

# Animation Start
t = threading.Thread(target=animate)
t.start()

clt = get_clt(image, color_count)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# Animation End
time.sleep(10)
done = True
 
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()