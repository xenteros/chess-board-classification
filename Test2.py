import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

img = cv2.imread('Szachownica2.jpg')
img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
print(len(img))
clt = KMeans(n_clusters=5) #cluster number
clt.fit(img)

hist = find_histogram(clt)
print(hist, clt.cluster_centers_)
bar = plot_colors2(hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()


# img = cv2.imread('Szachownica2.jpg')
# img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
# cv2.imshow("original", img)
#
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.imwrite('2.png',im2)
# cv2.drawContours(img, contours, -1, (0,255,0), 3)
# cv2.imwrite('3.png',img)
#
# cv2.imshow("contours", img)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()